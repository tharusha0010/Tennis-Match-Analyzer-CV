from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="rWJd7YhKrAjVu4Z395W7" 
)

result = CLIENT.infer(r"D:\Projects\Tennis-Match-Analyzer-CV\extracted_frames\frame_0046.jpg", model_id="tennis-match-analyzer-cv/2")

print(result)