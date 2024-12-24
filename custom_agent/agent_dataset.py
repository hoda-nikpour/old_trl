import json
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding

prompts = {
    "generate_query": "Please generate some mutually exclusive queries in a list to search the relevant papers according to the User Query. Searching for survey papers would be better.\nUser Query: {user_query}",
    "select_section": "You are conducting research on `{user_query}`. You need to predict which sections to look at for getting more relevant papers. Title: {title}\nAbstract: {abstract}\nSections: {sections}"
}

class AgentDataset(Dataset):
    def __init__(self, annotations_file, tokenizer):
        self.ids       = []
        self.messages  = []
        self.input_ids = []
        self.answers   = []
        self.lengths   = []
        with open(annotations_file) as f:
            for line in f.readlines():
                data = json.loads(line)
                prompt_template = data.get("prompt", "generate_query")
                if prompt_template == "generate_query":
                    prompt = prompts["generate_query"].format(user_query=data["user_query"])
                else:
                    prompt = prompts["select_section"].format(user_query=data["user_query"], title=data["title"], abstract=data["abstract"], sections=data["sections"])
                self.messages.append({
                    "content": prompt,
                    "role": "user"
                })
                self.ids.append(data["id"])
                input_ids = tokenizer.apply_chat_template(
                    [self.messages[-1]],
                    tokenize=True,
                    padding=True,
                    padding_side='left',
                    add_generation_prompt=True,
                )
                self.input_ids.append(input_ids)
                self.lengths.append(len(input_ids))
                answers_ids = tokenizer(
                    [json.dumps(data.get("answer", []))],
                    padding=True,
                    padding_side='left',
                )
                self.answers.append(answers_ids['input_ids'][0])
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "answers": self.answers[idx],
        }
    
    def __repr__(self):
        return "AgentDataset(\n    features: {},\n    length: {}\n)".format(json.dumps(list(self[0].keys())), len(self))


class AgentDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features):
        batch_input_ids = super().__call__([{"input_ids": feature["input_ids"]} for feature in features])
        batch_answers = super().__call__([{"input_ids": feature["answers"]} for feature in features])
        batch_input_ids["answers"] = batch_answers["input_ids"]
        del batch_input_ids["attention_mask"]
        return batch_input_ids
