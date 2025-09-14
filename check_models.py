import google.generativeai as genai

genai.configure(api_key="AIzaSyAaTIl966T8uHJePuZtDNhCPeOyQNWQeN8")

for model in genai.list_models():
   print(f"Model Name: {model.name}, Supported Methods: {model.supported_generation_methods}")
