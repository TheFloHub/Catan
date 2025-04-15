from openai import OpenAI
import base64
import json
from PIL import Image, ImageDraw

def detection_pipeline(answer: str, image_dir: str):
    try:
        answer = parse_json(answer)
        data = json.loads(answer)
        print("JSON detected")

        print(data)
        with Image.open(image_dir) as image:
            draw = ImageDraw.Draw(image)
            for obj in data:
                x0, y0, x1, y1 = obj["box_2d"]
                draw.rectangle([x0, y0, x1, y1], outline="red", width=4)
            image.show()
    except Exception as e:
        print("JSON not detected")
        print(e)


def parse_json(json_output):
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    return json_output


def read_file(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()
    raise Exception(f"Could not read file {file_path}")


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_text_content(text: str) -> dict:
    return {"type": "text", "text": text}


def get_image_content(image_path: str) -> dict:
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{encode_image(image_path)}",
        },
    }


def get_text_message(prompt: str) -> dict:
    return {
        "role": "user",
        "content": [
            get_text_content(prompt),
        ],
    }


def get_image_message(prompt: str, image_path: str) -> dict:
    return {
        "role": "user",
        "content": [
            get_text_content(prompt),
            get_image_content(image_path)
        ]
    }


def get_system_message() -> dict:
    return {
        "role": "system",
        "content": [
            get_text_content(read_file("game_rules.txt")),
            # get_text_content(
            #     "The next image shows from left to right a street, a settlement and a city for the blue player and the red player."),
            # get_image_content("images_system_prompt/figures.jpg"),
            
            get_text_content("A hexagonal tile representing a field which produces wheat. Its base color is yellow."),
            get_image_content("images_system_prompt/field.jpg"),
            
            get_text_content("A hexagonal tile representing a mountain which produces ore. Its base color is grey."),
            get_image_content("images_system_prompt/mountain.jpg"),
            
            get_text_content("A hexagonal tile representing a hill which produces brick. Its base color is brown."),
            get_image_content("images_system_prompt/hills.jpg"),
            
            get_text_content("A hexagonal tile representing a forest which produces wood. Its base color is dark green."),
            get_image_content("images_system_prompt/forest.jpg"),
            
            get_text_content("A hexagonal tile representing a pasture which produces sheep. Is bas color is light green."),
            get_image_content("images_system_prompt/pasture.jpg"),
        ],
    }


if __name__ == "__main__":
    # Client and model initialization
    openai_api_key = "your key"
    model_url = "your model url like chat gpt v1"
    client = OpenAI(api_key=openai_api_key, base_url=model_url)
    models = client.models.list()
    for model in models:
        print(model.id)
    model = "gemma3:27b-it-q8_0"
    if model is None:
        model = models.data[0].id
    print(f"Using model {model}")

    # Setup messages and completion

    # Example 1: Question about the board.
    # Some possible questions
    # How many Hills, Forests, Mountains, Fields and Pasture are on the board?
    # Which numbers lie on the mountain tiles?
    # How many roads, settlements and cities has the blue and red player?
    # How many forest, mountain, field, hill and pasture tiles are on the board?
    # How many 4s are there? Which resource tiles do they occupy?

    messages = [
        get_system_message(), 
        get_image_message("How many Hills, Forests, Mountains, Fields and Pasture are on the board?", "images_scenarios/20250401_121453.jpg"),
        ]
    
    # Example 2: Question about the progress of the game.
    # messages = [get_system_message(), 
    #     get_image_message("This is round 1.", "images_scenarios/20250401_121346.jpg"),
    #     get_image_message("This is round 2.", "images_scenarios/20250401_121854.jpg"),
    #     get_text_message("What happened between the two rounds?")
    #     ]

    # Example 3: Question about the board layout with one shot promtping by giving an example.
    # messages = [get_system_message(), get_image_message("M P F \n H F H W \n F P D P M \n H W M W \n F W P", "rotated/20250402_161910.jpg"), 
    #             get_image_message("Convert this image into the same text format.", "rotated/20250402_162002.jpg")]
    
    # Example 4: Localization of objects in JSON format.
    # img = "images_scenarios/20250401_121453.jpg"
    # messages = [
    #     get_system_message(), 
    #     get_image_message("Locate the mountain tiles, report the bounding box coordinates in JSON format (use box_2d).", img),
    #     ]


    # Inference
    completion = client.chat.completions.create(model=model, messages=messages)
    print(completion.choices[0].message.content)

    # Localization pipeline if JSON is detected
    if "img" in locals():
        detection_pipeline(completion.choices[0].message.content, img)