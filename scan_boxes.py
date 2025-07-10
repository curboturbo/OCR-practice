import json
from ultralytics import YOLO
import cv2
import numpy as np
import os
from dotenv import load_dotenv
import google.generativeai as genai

"""РАСПОЗНОВАНИЕ ЗАДАНИЯ И ОТНОШЕНИЕ К КОНКРЕТНОМУ НОМЕРУ, ПОДЛКЮЧЕНИЕ ВНЕШНЕЙ LLM"""


class Recognize:
    def __init__(self, json_file_path, photo, model, sensitivity) -> None:
        self.json_file_path: str = json_file_path
        self.photo: str = photo
        self.model: str = model
        self.sensitivity: float = sensitivity

    def check(self, x1_small, y1_small, x2_small, y2_small, x1_big, y1_big, x2_big, y2_big) -> bool:
        if (x1_small >= x1_big and
                y1_small >= y1_big and
                x2_small <= x2_big and
                y2_small <= y2_big):
            return True
        else:
            return False

    def extract_word_coordinates(self, json_data: json) -> dict:
        all_words_data = []
        for block in json_data.get('data', {}).get('blocks', []):
            for box in block.get('boxes', []):
                for lang_info in box.get('languages', []):
                    for text_segment in lang_info.get('texts', []):
                        for word_data in text_segment.get('words', []):
                            word = word_data.get('word')
                            x = word_data.get('x')
                            y = word_data.get('y')
                            w = word_data.get('w')
                            h = word_data.get('h')

                            if all(v is not None for v in [word, x, y, w, h]):
                                x2 = x + w
                                y2 = y + h
                                all_words_data.append({
                                    'word': word,
                                    'x1': x,
                                    'y1': y,
                                    'x2': x2,
                                    'y2': y2,
                                    'width': w,
                                    'height': h
                                })
        return all_words_data

    def main_function(self) -> dict:
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            word_coords = self.extract_word_coordinates(data)

        answer = dict()
        model = YOLO(self.model)
        class_names = model.names
        original_image = cv2.imread(self.photo)
        results = model(self.photo, save=False, conf=self.sensitivity)

        if not results or not hasattr(results[0], 'boxes') or len(results[0].boxes) == 0:
            # print("YOLO не обнаружил никаких объектов на изображении.")
            pass
        else:
            res = results[0]
            for i, box in enumerate(res.boxes):
                x1_yolo, y1_yolo, x2_yolo, y2_yolo = map(int, box.xyxy[0].tolist())
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown_Class_{class_id}"

                # print(f"\n--- Объект YOLO №{i+1}: {class_name} ---")
                # print(f"Координаты YOLO бокса: ({x1_yolo}, {y1_yolo}, {x2_yolo}, {y2_yolo}), Уверенность: {confidence:.2f}")
                # print("Найденные слова внутри этого бокса:")
                ans = ''
                c = 0
                for word_info in word_coords:
                    x1_word = word_info['x1']
                    y1_word = word_info['y1']
                    x2_word = word_info['x2']
                    y2_word = word_info['y2']
                    if self.check(x1_word, y1_word, x2_word, y2_word, x1_yolo, y1_yolo, x2_yolo, y2_yolo):
                        ans += ' ' + word_info['word']
                        c += 1

                if class_name in answer:
                    answer[f'{class_name}'] += ans
                else:
                    answer[f'{class_name}'] = ans
        results[0].show()
        return answer

    def __repr__(self):
        return 'Will be soon...'


class Gemini:
    def __init__(self) -> None:
        load_dotenv()
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    def send_to_check(self, message: str, task: str) -> str:
        genai.configure(api_key=self.GOOGLE_API_KEY)
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        prompt_parts = [
            f'проверь задание по биологии номер {task}, оцени и пришли мне четко оценку и комментарии двумя абзацами без лишнего (по шкале от 0 до 3) (обяательно пиши по русски)',
            message
        ]
        response = model.generate_content(prompt_parts)
        return response.text

    def correct_grammar(self, message: str) -> str:
        genai.configure(api_key=self.GOOGLE_API_KEY)
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        prompt_parts = [
            f'сделай грамматическое исправление в контексте того, что это задание ЕГЭ биологии',
            message
        ]
        response = model.generate_content(prompt_parts)
        return response.text

    def send_with_stipulation(self, message: str, stipulation: str, task: str) -> str:
        genai.configure(api_key=self.GOOGLE_API_KEY)
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        prompt_parts = [
            f'проверь задание по биологии номер {task}, оцени и пришли мне четко оценку и комментарии двумя абзацами без лишнего (по шкале от 0 до 3) (обяательно пиши по русски), вот улсовие задания : {stipulation}',
            message
        ]
        response = model.generate_content(prompt_parts)
        return response.text

    def send_with_discription(self, message: str, task: str) -> str:
        genai.configure(api_key=self.GOOGLE_API_KEY)
        with open(r"C:\Users\Tema\Desktop\criterion.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            benchmarks = data[f"{task}"]
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        prompt_parts = [
            f'проверь задание по биологии номер {task}, оцени и пришли мне четко оценку и комментарии двумя абзацами без лишнего (по шкале от 0 до 3) (обяательно пиши по русски), вот критерии проверки : {benchmarks}',
            message
        ]
        response = model.generate_content(prompt_parts)
        return response.text

    def __repr__(self):
        return 'Will be soon...'


def main(json_file_path, path, photo):
    a = Recognize(json_file_path, path, photo, 0.3).main_function()
    if a:
        for k, v in a.items():
            print(Gemini().send_with_discription(message=v, task=f'{k}'))
            print(
                '------------------------------------------------------------------------------------------------------------------------------\n')
    else:
        return 'Bounding box wasn t recognize'


"""json_file_path - path to discription of work,
path - path to model YOLOv8n,
photo - path to photo of student work"""
