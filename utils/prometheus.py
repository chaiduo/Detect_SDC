from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda:7"

model = AutoModelForCausalLM.from_pretrained("/data0/home/lc/cd/llm/prometheus/")
tokenizer = AutoTokenizer.from_pretrained("/data0/home/lc/cd/llm/prometheus/")

ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."

ABSOLUTE_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 0, and a score rubric representing an evaluation criteria are given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 0 and 2. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 0 and 2)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 0):
{reference_answer}

###Score Rubrics:
{rubric}

###Feedback: """

# 示例数据
sample_data = {
    "instruction": "What does the yellow and blue sign say?",
    "response": "The yellow and blue sign indicates a pedestrian crossing area.",
    "reference_answer": "The yellow and blue sign says \"Pedestrian Crossing.\"",
    "rubric": """Score 0: The answer is completely correct and matches the reference answer.
                 Score 1: The answer has minor deviations from the reference answer but is semantically correct.
                 Score 2: The answer is completely wrong or contains semantic errors."""
}

user_content = ABS_SYSTEM_PROMPT + "\n\n" + ABSOLUTE_PROMPT.format(**sample_data)

messages = [
    {"role": "user", "content": user_content},
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)

# 只解码新生成的部分（去掉输入的 prompt）
new_tokens = generated_ids[:, model_inputs['input_ids'].shape[1]:]
decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
print(decoded[0])