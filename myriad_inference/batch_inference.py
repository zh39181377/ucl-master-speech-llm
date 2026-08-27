import pandas as pd
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def generate_response(model, tokenizer, system_prompt, user_query):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    model_inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )
    
    if hasattr(model_inputs, "keys") or isinstance(model_inputs, dict):
        input_ids = model_inputs["input_ids"].to(model.device)
        attention_mask = model_inputs["attention_mask"].to(model.device)
    else:
        input_ids = model_inputs.to(model.device)
        attention_mask = torch.ones_like(input_ids).to(model.device)
    

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            max_new_tokens=512,
            temperature=0.7,       
            top_p=0.85,
            top_k=45,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False
        )
    
    input_length = input_ids.shape[1]
    response_ids = generated_ids[0][input_length:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    response = response.replace('Ċ', '\n').replace('Ġ', ' ').strip()
    
    return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True, help="deepseek model path with [MASK]")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA weigh path")
    parser.add_argument("--input_file", type=str, required=True, help="input excel path")
    parser.add_argument("--output_file", type=str, required=True, help="outpu excel path")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    
    print("Loading base model (in bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True
    )
    
    if args.lora_path:
        print(f"Injecting LoRA weights from {args.lora_path}...")
        model = PeftModel.from_pretrained(model, args.lora_path)
    
    model.eval()

    print(f"Reading dataset: {args.input_file}")
    df = pd.read_excel(args.input_file)
    predictions = []

    has_scenario = 'Scenario' in df.columns

    for index, row in df.iterrows():
        question = row['Masked_Question']
        

        if has_scenario and pd.notna(row['Scenario']):
            system_prompt = (
                f"You are a top-tier customer service expert strictly operating within the domain of: {row['Scenario']}. "
                "The user's input contains [MASK] tokens representing audio dropouts or noise. "
                "Your task is Direct Mapping: You must independently deduce the user's complete intent using your domain expertise "
                "and immediately provide the definitive, final answer. "
                "CRITICAL INSTRUCTIONS: Under NO circumstances should you ask follow-up questions, seek clarification, apologize, "
                "or state that the input is incomplete. Answer directly and confidently as if you heard the question perfectly."
            )
        else:
            system_prompt = (
                "You are an intuitive and highly professional customer service AI. "
                "The user's input contains [MASK] tokens simulating speech recognition errors. "
                "Your task is Direct Mapping: You must independently deduce the user's full intent using only the contextual clues "
                "from the available words and immediately provide the definitive, final answer. "
                "CRITICAL INSTRUCTIONS: Under NO circumstances should you ask follow-up questions, seek clarification, apologize, "
                "or state that the input is incomplete. Answer directly and confidently as if you heard the question perfectly."
            )
            
        ans = generate_response(model, tokenizer, system_prompt, question)
        predictions.append(ans)
        
        if (index + 1) % 10 == 0:
            print(f"Processed {index + 1}/{len(df)} samples...")

    df['Model_Response'] = predictions
    df.to_excel(args.output_file, index=False)
    print(f"Inference complete! Results saved to {args.output_file}")

if __name__ == "__main__":
    main()