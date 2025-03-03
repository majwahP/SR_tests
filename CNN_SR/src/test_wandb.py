import wandb

# ✅ Initialize WandB ONCE
wandb.init(
    project="test-project", 
    name="single-run2",
    config={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "optimizer": "Adam"
    }
)

# ✅ Log hyperparameters (already stored in `wandb.config`)
print("Using learning rate:", wandb.config.learning_rate)

# ✅ Log training metrics (accuracy & loss over 10 epochs)
for epoch in range(1, 11):  
    accuracy = 0.90 + epoch * 0.005  
    loss = 0.1 / epoch  

    wandb.log({
        "epoch": epoch,
        "accuracy": accuracy,
        "loss": loss
    })

# ✅ Log system performance at the end of training
wandb.log({
    "CPU Usage": wandb.util.cpu_load(),
    "RAM Usage": wandb.util.ram_usage()
})

# ✅ Finish the run
wandb.finish()

