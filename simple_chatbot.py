import openai
openai.api_key = "**********"

def Ans_ChatGPT(question):
    messages = [{
            "role": "user",
            "content": question,
        }]
    
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    response = completion.choices[0].message['content']
    return response

question = "hi"
answer = Ans_ChatGPT(question)

print(answer)
#Hello! How can I assist you today?
