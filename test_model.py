from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="your API key"
)

result = CLIENT.infer(r"D:\Projects\Tennis-Match-Analyzer-CV\extracted_frames\frame_0046.jpg", model_id="Your model ID")

print(result)