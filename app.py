from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import pandas as pd
import os
import re
import time
from rouge_score import rouge_scorer

app = Flask(__name__)
CORS(app)

# Load model dan tokenizer
model_path = "C:/Users/vin/Documents/skripsi/program/results"
tokenizer_path = "C:/Users/vin/Documents/skripsi/program/results"
tokenizer = BartTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
model = BartForConditionalGeneration.from_pretrained(model_path, local_files_only=True)

# Kata kunci tim dan entitas esports
kata_kunci_esports = [
    "esports", "gaming", "valorant", "league of legends", "mobile legends", "dota 2", "pubg", "tekken",
    "moba", "fps", "game online", "tim esports", "pro player", "tim profesional", "apac", "predator league",
    "gamers", "game", "esport", "esports team", "esports player", "mpl", "esl", "esl pro league",
    "aura", "prx", "btr", "bigetron", "boom esports", "dewa united esports", "evos", "evos glory",
    "evos icon", "gpx", "gpx racing", "gaimin gladiators", "geek fam", "geek slate", "mbr esports",
    "onic", "onic esports", "rebellion esports", "rrq", "rrq hoshi", "rrq kazu", "rsg ph", "rsg sg",
    "rrq mika", "rrq sena", "rebellion zion", "rrq kaito", "siren esports", "team liquid", "team redline",
    "tnc pro team", "omega esport", "tundra esport", "talon", "zeta division"
]

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# Load berita dari JSONL
berita_list = []
with open("C:/Users/vin/Documents/skripsi/program/ujicoba.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        berita = json.loads(line)
        berita['judul_clean'] = tokenize(berita.get('judul', ''))
        berita['konten_clean'] = tokenize(berita.get('konten', ''))
        berita_list.append(berita)

def cari_berita_relevan(berita_list, pertanyaan):
    pertanyaan_tokens = tokenize(pertanyaan)
    pertanyaan_lower = pertanyaan.lower()

    ada_kata_kunci_esports = any(k in pertanyaan_lower for k in kata_kunci_esports)
    if not ada_kata_kunci_esports:
        return None

    entitas_pertanyaan = [token for token in pertanyaan_tokens if token in kata_kunci_esports]

    skor_berita = []
    for berita in berita_list:
        judul_tokens = berita.get('judul_clean', [])
        konten_tokens = berita.get('konten_clean', [])

        skor_judul = sum(1 for kata in pertanyaan_tokens if kata in judul_tokens)
        skor_konten = sum(1 for kata in pertanyaan_tokens if kata in konten_tokens)

        skor_total = 0.4 * skor_judul + 0.6 * skor_konten

        bonus_entitas = sum(1 for entitas in entitas_pertanyaan if entitas in judul_tokens or entitas in konten_tokens)
        skor_total += 0.5 * bonus_entitas

        if skor_konten >= 2 or bonus_entitas > 0:
            skor_berita.append((skor_total, berita))

    if skor_berita:
        skor_berita.sort(reverse=True, key=lambda x: x[0])
        return skor_berita[0][1]

    return None

# Evaluasi ROUGE-L
def evaluasi_rouge(pred, ref):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(ref, pred)
    rougeL = scores['rougeL']
    return rougeL.fmeasure, rougeL.precision, rougeL.recall

# Evaluasi CBE
def evaluasi_cbe(pertanyaan, jawaban):
    relevansi = 4 if any(kata in jawaban.lower() for kata in pertanyaan.lower().split()) else 2
    koherensi = 5 if len(jawaban.split()) > 10 else 3
    fluency = 5 if jawaban[0].isupper() and jawaban.endswith('.') else 3
    return relevansi, koherensi, fluency

# Hitung TP, FP, TN, FN
def hitung_confusion_metrics(pred, ref):
    pred_tokens = set(tokenize(pred))
    ref_tokens = set(tokenize(ref))

    tp = len(pred_tokens & ref_tokens)
    fp = len(pred_tokens - ref_tokens)
    fn = len(ref_tokens - pred_tokens)
    tn = 0  # Tidak dapat dihitung tanpa korpus negatif

    return tp, fp, tn, fn

@app.route('/chat', methods=['POST'])
def chat():
    try:
        start_time = time.time()

        data = request.get_json()
        user_input = data.get('question')

        if not user_input:
            return jsonify({"error": "Pertanyaan kosong."}), 400

        berita = cari_berita_relevan(berita_list, user_input)

        if berita:
            context = berita.get('konten', '').replace('\n', ' ').strip()
            url = berita.get('url', '')
            ground_truth = berita.get('konten', '').strip()
            prompt = f"\n\n\n{context}"
        else:
            prompt = f"\nTidak ada berita relevan."
            url = ""
            ground_truth = ""

        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        output_ids = model.generate(
            **inputs,
            max_length=400,
            num_beams=6,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        sentences = answer.split('.')
        if len(sentences) > 3:
            answer = '.'.join(sentences[:3]).strip() + '.'
        else:
            answer = answer.strip()

        if ground_truth:
            f1, precision, recall = evaluasi_rouge(answer, ground_truth)
            tp, fp, tn, fn = hitung_confusion_metrics(answer, ground_truth)
        else:
            f1 = precision = recall = None
            tp = fp = tn = fn = None

        relevansi, koherensi, fluency = evaluasi_cbe(user_input, answer)

        response_time = round(time.time() - start_time, 3)

        log_file = "chat_log.xlsx"
        log_data = {
            "pertanyaan": [user_input],
            "jawaban_model": [answer],
            "f1_score_rougeL": [f1],
            "precision_rougeL": [precision],
            "recall_rougeL": [recall],
            "tp": [tp],
            "fp": [fp],
            "tn": [tn],
            "fn": [fn],
            "relevansi": [relevansi],
            "koherensi": [koherensi],
            "fluency": [fluency],
            "url": [url],
            "response_time_sec": [response_time]
        }

        if os.path.exists(log_file):
            df = pd.read_excel(log_file)
            df = pd.concat([df, pd.DataFrame(log_data)], ignore_index=True)
        else:
            df = pd.DataFrame(log_data)

        df.to_excel(log_file, index=False)

        response = {
            "answer": answer,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "relevansi": relevansi,
            "koherensi": koherensi,
            "fluency": fluency
        }

        if url:
            response["url"] = url

        return jsonify(response)

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Terjadi kesalahan di server."}), 500

if __name__ == "__main__":
    app.run(debug=True)
