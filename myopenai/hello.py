

import openai
def printHello():
    openai.api_key = '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'
    
    message = "請問哪一國家贏得2018年奧林匹科籃球賽冠軍"
    print(message)
    messages = [
                   {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge"},
                   {"role": "assistant", "content" : "I am doing well"},
                   {"role": "user", "content": message},
                   ]
    response = openai.ChatCompletion.create(
                model ="gpt-3.5-turbo",
               #prompt=msg,
                messages = messages,
               #[
               #        {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible."},
               #        {"role": "user", "content": message},
               #          ],
                max_tokens=128,
                temperature=0
            )


    completed_text = response["choices"][0]["message"]["content"]
    return (completed_text)



