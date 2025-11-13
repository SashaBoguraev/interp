import torch, os, argparse, datetime, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyvene import set_seed

from interventions.das import run_das_experiment
from utils import DEFAULT_CONFIG, neural_interventions


def eval_behavior(model, tokenizer, evalset, device):
    pass  # Placeholder for actual evaluation logic


def get_data(tokenizer, total_samples, batch_size):
    pass  # Placeholder for actual data loading logic


def main(model, target_layer, stream, total_samples, intervention_pos, batch_size, epochs, low_rank_dimension, learning_rate, gradient_accumulation_steps, \
         intervention_types, eval_factual, num_layers, hidden_dim, seed, cache_dir, device, wandb):

    # Set seed for reproducibility
    set_seed(seed)
    
    # Create output directory with intervention type
    model_name = model.split("/")[-1].lower()
    output_dir = f"results/"

    print(f"Loading model {model}...")
    # Load model and tokenizer matching the notebook
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model, cache_dir=cache_dir)
    _ = model.to(device)
    _ = model.eval()  # always no grad on the model

    # Create datasets
    trainset, evalset = get_data(
        tokenizer, total_samples, batch_size
    )


    if eval_factual or not DEFAULT_CONFIG["baseline_accuracy"]:
        factual_acc = eval_behavior(model, tokenizer, evalset, device)
    else:
        factual_acc = DEFAULT_CONFIG["baseline_accuracy"]  # Use predefined baseline if available
    

    _ = model.to("cpu")
    del model
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    _ = model.to(device)
    _ = model.eval()  # always no grad on the model


    extra_id = f'_nlayers{num_layers}_hiddendim{hidden_dim}' if intervention_type in neural_interventions else ''
    NOW = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output=[]
    for intervention_type in intervention_types:
        result = run_das_experiment(
            intervention_name=intervention_type,
            model=model,
            tokenizer=tokenizer,
            layer=target_layer,
            component=stream,
            train_dataloader=trainset,
            test_dataloader=evalset,
            intervention_pos=intervention_pos,
            epochs=epochs,
            low_rank_dimension=low_rank_dimension,
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps,
            device=device,
            num_layers=num_layers,
            hidden_dim=hidden_dim
            ,
            # Pass wandb settings via config in run_das_experiment
            config={
                "use_wandb": wandb,
                "wandb_project": 'nlf-das',
                "wandb_run_name": f"{intervention_type}_layer{target_layer}{extra_id}_{seed}",
                "wandb_run_id": f"{intervention_type}_layer{target_layer}{extra_id}_{seed}_{NOW}",
            }
        )
        accuracy = result['accuracy']
        output.append({
            "layer": target_layer,
            "accuracy": accuracy,
            "baseline_accuracy": factual_acc,
            "improvement": accuracy - factual_acc,
            "intervention_type": intervention_type,
            "model": model,
            "num_layers": num_layers if intervention_type in neural_interventions else None,
            "hidden_dim": hidden_dim if intervention_type in neural_interventions else None
        })
    
    output_df = pd.DataFrame(output)
    output_path = os.path.join(output_dir+f'/raw/{model_name}', f"experiment_results_layer{target_layer}_{NOW}.csv")
    output_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="DAS Experiment")
    
    # Basic arguments
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model to use for training")
    parser.add_argument("--intervention_types", type=str, nargs='+', default=['low_rank'],
                        choices=["das", "low_rank", "boundless", "stretch", "realnvp", "revnet", "unconstrained"],
                        help="Type of intervention to apply")
    
    # Training parameters
    parser.add_argument("--stream", type=str, default="block_output")
    parser.add_argument("--target_layer", type=int, default=DEFAULT_CONFIG["target_layer"],
                        help="Layer to target if not doing search")
    parser.add_argument("--intervention_pos", type=int, default=DEFAULT_CONFIG["intervention_pos"],
                        help="Position to apply the intervention")
    parser.add_argument("--low_rank_dimension", type=int, default=DEFAULT_CONFIG["low_rank_dimension"],
                        help="Low rank dimension for low_rank intervention")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"],
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"],
                        help="Batch size for training")
    parser.add_argument("--total_samples", type=int, default=DEFAULT_CONFIG["total_samples"],
                        help="Total number of samples to generate")
    parser.add_argument("--learning_rate", "-lr", type=float, default=DEFAULT_CONFIG["learning_rate"],
                        help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, 
                        default=DEFAULT_CONFIG["gradient_accumulation_steps"],
                        help="Gradient accumulation steps")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers for the intervention")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension for the intervention")

    # Optional wandb integration
    parser.add_argument("--wandb", "-wb", action="store_true", default=False, help="Enable logging to Weights & Biases")

    # Misc parameters
    parser.add_argument("--output_dir", type=str, default="results/",
                        help="Directory to save results")
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CONFIG["cache_dir"],
                        help="Cache directory for models")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--eval_factual", action="store_true",
                        help="Evaluate factual performance first")

    
    args = parser.parse_args()
    args.wandb = not args.wandb # Flag should turn it off

    main(
        model=args.model,
        target_layer=args.target_layer,
        stream='all',
        total_samples=args.total_samples,
        intervention_pos=args.intervention_pos,
        batch_size=args.batch_size,
        epochs=args.epochs,
        low_rank_dimension=args.low_rank_dimension,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        intervention_types=[args.intervention_type],
        eval_factual=args.eval_factual,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        cache_dir=args.cache_dir,
        device=args.device,
        wandb=args.wandb
    )