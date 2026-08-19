from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="5tedwy8cTM2JfRId8RHu" 
)

result = CLIENT.infer(r"D:\Projects\Tennis-Match-Analyzer-CV\extracted_frames\frame_0046.jpg", model_id="tennis-match-analyzer-cv-bbwo3/1")

print(result)