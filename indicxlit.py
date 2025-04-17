import json
import re
import torch
import inflect
import wandb
from ai4bharat.transliteration import XlitEngine
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from evaluate import load

wandb.init(project="sarvam-inference-pipeline", name="text_normalization_pipeline", entity="abhavana0410-ssn-coe")

sarvam_model_name = "sarvamai/sarvam-1"
sarvam_tokenizer = AutoTokenizer.from_pretrained(sarvam_model_name)
sarvam_model = AutoModelForSequenceClassification.from_pretrained(sarvam_model_name)

inflect_engine = inflect.engine()
xlit_engine = None

def convert_number_to_words(number):
    return inflect_engine.number_to_words(number)

def convert_year_to_words(year):
    if 1000 <= int(year) <= 9999:
        first_part = int(year[:2])
        second_part = int(year[2:])
        return f"{inflect_engine.number_to_words(first_part)} {inflect_engine.number_to_words(second_part)}"
    return inflect_engine.number_to_words(year)

def normalize_dates(text):
    date_patterns = [
        r"(\d{2})/(\d{2})/(\d{4})",
        r"(\d{4})-(\d{2})-(\d{2})",
        r"(\d{2})-(\d{2})-(\d{4})",
        r"([A-Za-z]+) (\d{1,2}), (\d{4})"
    ]

    def date_replacer(match):
        day, month, year = match.groups()
        if len(day) > 2:
            year, month, day = day, month, year
        month_name = ["January", "February", "March", "April", "May", "June",
                      "July", "August", "September", "October", "November", "December"][int(month) - 1]
        return f"{convert_number_to_words(day)} {month_name} {convert_year_to_words(year)}"

    for pattern in date_patterns:
        text = re.sub(pattern, date_replacer, text)
    return text

def normalize_currency(text):
    currency_patterns = [
        (r"\$(\d+)", "dollars"), (r"\u20B9(\d+)", "rupees"), (r"\u20AC(\d+)", "euros"),
        (r"\u00A3(\d+)", "pounds"), (r"\u00A5(\d+)", "yen"), (r"(\d+)\s?₩", "won"),
        (r"(\d+)\s?₽", "rubles"), (r"(\d+)\s?﷼", "riyals"), (r"(\d+)\s?₫", "dong"), (r"(\d+)\s?₺", "lira")
    ]
    def currency_replacer(match, unit):
        amount = match.group(1)
        return f"{convert_number_to_words(amount)} {unit}"
    for pattern, unit in currency_patterns:
        text = re.sub(pattern, lambda m: currency_replacer(m, unit), text)
    return text

def normalize_units(text):
    unit_pattern = r"(\d+)(mm|cm|m|km|in|ft|yd|mi|mg|kg|g|ml|L|gal|oz|°C|°F|J|kJ|W|kW|Pa|kPa|N|Hz|A|V)"
    unit_map = {
        "mm": "millimeters", "cm": "centimeters", "m": "meters", "km": "kilometers",
        "in": "inches", "ft": "feet", "yd": "yards", "mi": "miles", "mg": "milligrams",
        "kg": "kilograms", "g": "grams", "mg": "milligrams", "ml": "milliliters", "L": "liters", "gal": "gallons",
        "oz": "ounces", "°C": "degrees Celsius", "°F": "degrees Fahrenheit", "J": "joules",
        "kJ": "kilojoules", "W": "watts", "kW": "kilowatts", "Pa": "pascals", "kPa": "kilopascals",
        "N": "newtons", "Hz": "hertz", "A": "amperes", "V": "volts"
    }
    def unit_replacer(match):
        value, unit = match.groups()
        return f"{convert_number_to_words(value)} {unit_map[unit]}"
    return re.sub(unit_pattern, unit_replacer, text)

def normalize_text(text):
    text = normalize_dates(text)
    text = normalize_currency(text)
    text = normalize_units(text)
    return text

def detect_language_from_entry(entry):
    return entry.get("language", "hi")

def transliterate_text(text, lang_code):
    global xlit_engine
    xlit_engine = XlitEngine(lang_code, beam_width=10)
    return xlit_engine.translit_sentence(text)

def process_text(input_text, lang_code):
    normalized_text = normalize_text(input_text)
    transliterated_text = transliterate_text(normalized_text, lang_code)
    return transliterated_text

input_file = "transliteration_dataset.json"
output_file = "output_results.json"

with open(input_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)

wer = load("wer")
chrf = load("chrf")

results = []
for entry in dataset:
    input_text = entry["input"]
    expected_output = entry["expected_output"]
    lang_code = detect_language_from_entry(entry)
    generated_output = process_text(input_text, lang_code)
    if isinstance(generated_output, dict):
        generated_output = generated_output.get(lang_code, "")
    results.append({
        "input": input_text,
        "generated_output": generated_output,
        "expected_output": expected_output
    })

    wer.add_batch(predictions=[generated_output], references=[expected_output])
    chrf.add_batch(predictions=[generated_output], references=[expected_output])

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("WER:", wer.compute())
print("CHRF:", chrf.compute()["score"])
wandb.finish()
