import json
import torch
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("data/plain_sql.json", "r") as file:
    data_str = file.read()  # Read the content of the file as a string
    data = json.loads(data_str)  # Parse the JSON string

prompts = []
labels = []

# Iterate over the data
for repo_url, commits_info in data.items():
    for commit_hash, commit_info in commits_info.items():
        # Check if the 'files' key exists in the commit_info
        for file_info in commit_info["files"].values():
            for change in file_info["changes"]:
                prompt_lines = change["diff"].split("\n- ")[1:]
                label_lines = change["diff"].split("\n+ ")[1:]

                for prompt_line, label_line in zip(prompt_lines, label_lines):
                    prompt = prompt_line.split("\n")[0].strip()
                    label = label_line.split("\n")[0].strip()
                    prompts.append(prompt)
                    labels.append(label)

# Choose a valid model identifier
model_identifier = "bert-base-uncased"

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_identifier)

# Tokenize prompts and labels
prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
label_inputs = tokenizer(labels, return_tensors="pt", padding=True, truncation=True)

# Check the tokenized batch sizes
print("Prompt input batch size:", prompt_inputs["input_ids"].shape[0])
print("Label input batch size:", label_inputs["input_ids"].shape[0])



#INSTALL_DIR=$HOME/cuda-gdb-darwin-12.3
#PATH=$INSTALL_DIR/bin:$PATH
#CUDA_HOME=$INSTALL_DIR/bin:$PATH
#pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html




model_name = "petals-team/StableBeluga2"
# You can also use any other supported model from 🤗 Model Hub

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, add_bos_token=False)
model = AutoDistributedModelForCausalLM.from_pretrained(model_name)
#model = model.cuda()

inputs = tokenizer("WHERE parent_id IN ({list_root_ids}", return_tensors="pt")["input_ids"]
outputs = model.generate(inputs, max_new_tokens=3)
print(tokenizer.decode(outputs[0]))



# Fine-tune the model
for prompt, label in zip(prompts, labels):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    label_ids = tokenizer(label, return_tensors="pt")["input_ids"]

    # Ensure the batch sizes match
    max_batch_size = max(input_ids.shape[1], label_ids.shape[1])
    padded_input_ids = torch.nn.functional.pad(input_ids, (0, max_batch_size - input_ids.shape[1]))
    padded_label_ids = torch.nn.functional.pad(label_ids, (0, max_batch_size - label_ids.shape[1]))

    # Convert tensors to floating point type
    input_ids = input_ids.float()
    label_ids = label_ids.float()
    # Enable gradients for the tensors
    input_ids.requires_grad_()
    label_ids.requires_grad_()

    print("Batch size of padded_input_ids:", padded_input_ids.shape[1])
    print("Batch size of padded_label_ids:", padded_label_ids.shape[1])

    # Move tensors to GPU if available
    input_ids = input_ids.to(device)
    label_ids = label_ids.to(device)

    for i in range(12):  # You may adjust the number of iterations
        loss = model(input_ids=padded_input_ids, labels=padded_label_ids).loss
        print(f"Loss[{i}] = {loss.item():.3f}")

        model.zero_grad()
        loss.backward()
        model.optimizer.step()
        print("Model optimizer step")



