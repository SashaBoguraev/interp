import torch
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup
from pyvene import (
    IntervenableModel,
    set_seed,
    count_parameters,
)

from interventions.configs import get_config
from interventions.metrics import compute_metrics, calculate_loss
import gc
from transforms.utils import create_projection_model
import wandb


def run_das_experiment(
    intervention_name,
    model,
    tokenizer,
    layer,
    component="block_output",
    train_dataloader=None,
    test_dataloader=None,
    intervention_pos=60,
    epochs=1,
    low_rank_dimension=1,
    learning_rate=1e-3,
    temperature_scheduling=False,
    gradient_accumulation_steps=4,
    transformation=None,
    num_layers=None,
    hidden_dim=None,
    config=None,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Run DAS experiment matching the notebook's approach.
    
    Args:
        intervention_name: 'low_rank', 'boundless', or 'das'
        model: The language model
        tokenizer: The tokenizer
        layer: Layer number for intervention
        component: Component type (default: "block_output")
        train_dataloader: Training data
        test_dataloader: Test data
        intervention_pos: Position to apply intervention (default: 60)
        epochs: Number of training epochs (default: 1)
        low_rank_dimension: Dimension for low rank interventions (default: 1)
        learning_rate: Learning rate (default: 1e-3)
        temperature_scheduling: Whether to use temperature scheduling (default: True)
        gradient_accumulation_steps: Gradient accumulation steps (default: 4),
        transformation: Optional transformation to apply (e.g., 'cubic', 'translation')
    """
    print(f"\nRunning {intervention_name} experiment on layer {layer}, position {intervention_pos}")
    
    # Set seed for reproducibility
    set_seed(42)
    
    if transformation:
        model = create_projection_model(model, transformation, layer)
        intervention_layer = layer + 1
        print(f"  Inserted transformation layers at positions {layer+1} (forward) and {layer+2} (inverse)")
        print(f"  Intervening at layer {intervention_layer} (after transformation round-trip)")
    else:
        intervention_layer = layer

    print(f"  Intervention layer set to {intervention_layer}")

    # Optional wandb integration
    use_wandb = config.get("use_wandb", False) if isinstance(config, dict) else False
    if use_wandb:
        wandb.login()
        run = wandb.init(
            project=config.get("wandb_project", "das-experiments"),
            name=config.get("wandb_run_name", None) if isinstance(config, dict) else None,
            id=config.get("wandb_run_id", None) if isinstance(config, dict) else None,
            config={
                "intervention_name": intervention_name,
                "intervention_layer": intervention_layer,
                "component": component,
                "low_rank_dimension": low_rank_dimension,
                "num_layers": num_layers,
                "hidden_dim": hidden_dim,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "temperature_scheduling": temperature_scheduling,
                "task": "price-tagging",
                "transformation": transformation,
                "seed": 42
            }
        )

    # Force clean model state
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Create configuration matching the notebook
    config, layer_type = get_config(
        intervention_name=intervention_name,
        model_type=type(model),
        layer=intervention_layer,
        component=component,
        low_rank_dimension=low_rank_dimension,
        n_layers=num_layers,
        hidden_dim=hidden_dim,
        latent_dim=model.config.hidden_size
    )
    
    # Create intervenable model
    intervenable = IntervenableModel(config, model)
    intervenable.set_device(device)
    intervenable.disable_model_gradients()
    
    # Initialize gradients
    intervenable.set_zero_grad()

    # Setup optimizer exactly like the notebook
    t_total = int(len(train_dataloader) * epochs)
    warm_up_steps = 0.1 * t_total
    optimizer_params = []
    
    for k, v in intervenable.interventions.items():
        layer = getattr(v, layer_type)
        optimizer_params += [{"params": layer.parameters()}]
        # Add boundary parameters for boundless interventions
        if intervention_name == "boundless":
            optimizer_params += [{"params": v.intervention_boundaries, "lr": 1e-2}]
    
    optimizer = torch.optim.Adam(optimizer_params, lr=learning_rate)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warm_up_steps, num_training_steps=t_total
    )

    # Temperature scheduling setup
    total_step = 0
    target_total_step = len(train_dataloader) * epochs
    
    if temperature_scheduling:
        temperature_start = 50.0
        temperature_end = 0.1
        temperature_schedule = (
            torch.linspace(temperature_start, temperature_end, target_total_step)
            .to(torch.bfloat16)
            .to(device)
        )
        intervenable.set_temperature(temperature_schedule[total_step])

    # Training history
    # loss_history = []
    # accuracy_history = []

    # Training loop matching the notebook
    intervenable.model.train()  # train enables drop-off but no grads
    print("llama trainable parameters: ", count_parameters(intervenable.model))
    print("intervention trainable parameters: ", intervenable.count_parameters())
    
    train_iterator = trange(0, int(epochs), desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc=f"Epoch: {epoch}", position=0, leave=True
        )
        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            
            b_s = inputs["input_ids"].shape[0]
            _, counterfactual_outputs = intervenable(
                {"input_ids": inputs["input_ids"]},
                [{"input_ids": inputs["source_input_ids"]}],
                {"sources->base": intervention_pos},  # swap pos-th token
            )
            
            eval_metrics = compute_metrics(
                [counterfactual_outputs.logits], [inputs["labels"]]
            )

            # Loss and backprop - matching notebook's calculate_loss
            loss = calculate_loss(counterfactual_outputs.logits, inputs["labels"], intervenable, intervention_name)
            loss_str = round(loss.item(), 2)
            epoch_iterator.set_postfix({"loss": loss_str, "acc": eval_metrics["accuracy"]})

            # Log to wandb if available
            if wandb is not None:
                try:
                    run.log({
                        "train/loss": loss.item(),
                        "train/accuracy": eval_metrics["accuracy"],
                        "train/epoch": epoch,
                        "train/step": total_step
                    }, step=total_step)
                except Exception:
                    # Don't fail training for wandb issues
                    pass

            # loss_history.append(loss_str)
            # accuracy_history.append(eval_metrics["accuracy"])

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            
            if total_step % gradient_accumulation_steps == 0:
                if not (gradient_accumulation_steps > 1 and total_step == 0):
                    optimizer.step()
                    scheduler.step()
                    intervenable.set_zero_grad()
                    if temperature_scheduling:
                        intervenable.set_temperature(temperature_schedule[total_step])
            
            # Clean up memory after each training step
            del counterfactual_outputs, loss
            
            total_step += 1

    torch.cuda.empty_cache()
    gc.collect()
    intervenable.model.eval()  # eval disables drop-off

    # Evaluation loop matching the notebook
    eval_labels = []
    eval_preds = []
    with torch.no_grad():
        epoch_iterator = tqdm(test_dataloader, desc=f"Test")
        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            
            b_s = inputs["input_ids"].shape[0]
            _, counterfactual_outputs = intervenable(
                {"input_ids": inputs["input_ids"]},
                [{"input_ids": inputs["source_input_ids"]}],
                {"sources->base": intervention_pos},  # swap intervention_pos-th token
            )
            eval_labels += [inputs["labels"].cpu()]
            eval_preds += [counterfactual_outputs.logits.cpu()]
            
            # Clear memory after each batch
            del counterfactual_outputs
            for k in list(inputs.keys()):
                if isinstance(inputs[k], torch.Tensor):
                    del inputs[k]
            
            if step % 10 == 0:  # periodic cleanup
                torch.cuda.empty_cache()
    
    eval_metrics = compute_metrics(eval_preds, eval_labels)
    print(f"Final test accuracy: {eval_metrics['accuracy']}")

    # Final wandb logging
    if wandb is not None:
        try:
            run.log({
                "eval/accuracy": eval_metrics["accuracy"],
            })
            # finish the run if we initialized it here
            if run is not None:
                run.finish()
        except Exception:
            pass
    
    return {
        "intervenable": intervenable,
        "accuracy": eval_metrics["accuracy"]
    }


# Legacy function for backward compatibility
def run_das_experiment_legacy(intervention, model, tokenizer, target_representation, train_dataloader, test_dataloader, intervention_pos, epochs=3, max_intervention_dimension=40):
    """
    Legacy wrapper for backward compatibility.
    """
    return run_das_experiment(
        intervention_name=intervention,
        model=model,
        tokenizer=tokenizer,
        layer=target_representation.layer,
        component=target_representation.component,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        intervention_pos=intervention_pos,
        epochs=epochs,
        low_rank_dimension=1
    )["accuracy"]