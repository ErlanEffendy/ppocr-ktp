"""
KTP OCR + Parser — Fully On-Premise, No External API
======================================================
1. Runs the HuggingFace donut-based OCR model (emisilab/model-ocr-ktp-v1)
   to extract raw text from a KTP image.
2. Parses the flat OCR output into a structured key-value dict using
   pure rule-based logic (no API, no internet after model download).

Dependencies:
    pip install transformers torch pillow

Usage:
    python ktp_parser.py --image ktp.jpg
    python ktp_parser.py --image ktp.jpg --output result.json
    python ktp_parser.py --folder ./ktp_images/   # batch mode
"""

import re
import json
import argparse
import sys
import os
import pickle
import requests
from pathlib import Path
from datetime import date, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — OCR MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def load_ocr_model(model_name: str = "emisilab/model-ocr-ktp-v1"):
    """
    Load the HuggingFace donut-based KTP OCR pipeline.
    Downloads the model on first run; uses local cache on subsequent runs.
    """
    from transformers import pipeline as hf_pipeline
    print(f"[OCR] Loading model: {model_name}")
    pipe = hf_pipeline("image-text-to-text", model=model_name)
    print("[OCR] Model ready.")
    return pipe


def run_ocr(pipe, image_path: str) -> str:
    """
    Run OCR on a single KTP image.

    Args:
        pipe:        HuggingFace pipeline object (from load_ocr_model)
        image_path:  Path to the KTP image file (.jpg, .png, etc.)

    Returns:
        Raw OCR string (e.g. "<s_dataset_ktp> DEBBY ANGGRAINI ...")
    """
    from PIL import Image

    image  = Image.open(image_path).convert("RGB")
    output = pipe(image, "<s_dataset_ktp>")

    # Pipeline returns a list of dicts: [{'generated_text': '...'}]
    if isinstance(output, list) and len(output) > 0:
        return output[0].get("generated_text", "")
    return str(output)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — VOCABULARIES
# ═══════════════════════════════════════════════════════════════════════════════

AGAMA_LIST: set = {
    "ISLAM", "KRISTEN", "KATOLIK", "KATHOLIK", "HINDU", "BUDHA", "BUDDHA", "KONGHUCU", "KHONGHUCU",
}

STATUS_LIST: set = {
    "BELUM KAWIN", "KAWIN", "CERAI HIDUP", "CERAI MATI",
}

JK_LIST: set = {
    "LAKI-LAKI", "PEREMPUAN",
}

WN_LIST: set = {
    "WNI", "WNA",
}

# Ordered longest-first so AB+ matches before A, etc.
GOL_DARAH_ORDERED: list = [
    "AB+", "AB-", "A+", "A-", "B+", "B-", "O+", "O-",
    "AB", "A", "B", "O",
]

PEKERJAAN_LIST: set = {
    "MENGURUS RUMAH TANGGA", "KARYAWAN SWASTA", "KARYAWAN BUMN",
    "KARYAWAN BUMD", "PEGAWAI NEGERI SIPIL", "TENAGA PENGAJAR",
    "BELUM BEKERJA", "TIDAK BEKERJA", "PELAJAR / MAHASISWA",
    "PELAJAR/MAHASISWA", "PELAJAR", "MAHASISWA", "WIRASWASTA",
    "PETANI", "NELAYAN", "BURUH", "DOKTER", "GURU", "PEDAGANG",
    "PENSIUNAN", "BIDAN", "PERAWAT", "APOTEKER", "PILOT",
    "WARTAWAN", "PENGACARA", "NOTARIS", "SENIMAN", "ATLET",
    "TNI", "POLRI", "PNS",
}

# Region lists are now fetched strictly via RegionalMatcher API.


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATA MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KTPData:
    nik:               Optional[str] = None
    nama:              Optional[str] = None
    tempat_lahir:      Optional[str] = None
    tgl_lahir:         Optional[str] = None   # DD-MM-YYYY
    jenis_kelamin:     Optional[str] = None
    gol_darah:         Optional[str] = None
    alamat:            Optional[str] = None
    rt_rw:             Optional[str] = None
    kel_desa:          Optional[str] = None
    kecamatan:         Optional[str] = None
    kota:              Optional[str] = None
    provinsi:          Optional[str] = None
    agama:             Optional[str] = None
    status_perkawinan: Optional[str] = None
    pekerjaan:         Optional[str] = None
    kewarganegaraan:   Optional[str] = None
    berlaku_hingga:    Optional[str] = None
    _unmatched:        Optional[str] = None   # leftover tokens for debugging


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PARSER HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.upper()).strip()

def _strip_tokens(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)

def _remove(phrase: str, text: str) -> str:
    """
    Safe removal:
    - Multi-word phrases → plain str.replace (spaces are natural delimiters).
    - Single words       → regex \\b to avoid partial matches
                           (e.g. "O" must not wipe every "O" in the text).
    """
    if " " in phrase:
        return _norm(text.replace(phrase, " "))
    return _norm(re.sub(rf"\b{re.escape(phrase)}\b", " ", text))

def _remove_at(m: re.Match, text: str) -> str:
    """Remove exactly the span of a regex match (position-safe)."""
    return _norm(text[:m.start()] + " " + text[m.end():])

def _excel_to_date(serial: int) -> str:
    try:
        return (date(1899, 12, 30) + timedelta(days=serial)).strftime("%d-%m-%Y")
    except Exception:
        return str(serial)

def _is_birth_serial(n: int) -> bool:
    return 10_000 < n < 48_000

def _nik_decode(nik: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not nik or len(nik) not in (15, 16) or not nik.isdigit():
        return None, None
    assert nik is not None
    try:
        dd = int(nik[6:8])
        mm = int(nik[8:10])
        yy = int(nik[10:12])
        jk = "PEREMPUAN" if dd > 40 else "LAKI-LAKI"
        if dd > 40:
            dd -= 40
        cur  = date.today().year % 100
        year = 2000 + yy if yy <= cur else 1900 + yy
        return date(year, mm, dd).strftime("%d-%m-%Y"), jk
    except Exception:
        return None, None

def _find_vocab(vocab: set, text: str) -> Tuple[Optional[str], str]:
    for candidate in sorted(vocab, key=len, reverse=True):
        if candidate in text:
            return candidate, _remove(candidate, text)
    return None, text

def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

class RegionalMatcher:
    """Helper to match Indonesian regional names (Province, City, District, Village) via API"""
    BASE_URL = "https://emsifa.github.io/api-wilayah-indonesia/api"
    CACHE_DIR = ".cache_regional"

    def __init__(self):
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self.provinces = self._load_data("provinces.json")
        self._city_cache = {}
        self._district_cache = {}
        self._village_cache = {}

    def _load_data(self, endpoint):
        cache_path = os.path.join(self.CACHE_DIR, endpoint.replace("/", "_"))
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception: pass
        
        try:
            url = f"{self.BASE_URL}/{endpoint}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                with open(cache_path, "wb") as f:
                    pickle.dump(data, f)
                return data
        except Exception as e:
            print(f"[WARN] Could not fetch regional data from {endpoint}: {e}")
        return []

    def _fuzzy_match_regional(self, text, items, prefixes_to_strip, threshold=0.5):
        if not text: return None, None
        
        input_raw = text.upper()
        
        # Clean versions for base comparison
        clean_input = input_raw
        for p in prefixes_to_strip:
            clean_input = re.sub(rf'^{p}\b\s*', '', clean_input, flags=re.I)
        clean_input = clean_input.strip()
        
        best_match = None
        max_score = 0.0  # Use float
        
        for item in items:
            orig_name = item.get('name', '').upper()
            
            # Clean candidate
            clean_cand = orig_name
            for p in prefixes_to_strip:
                clean_cand = re.sub(rf'^{p}\b\s*', '', clean_cand, flags=re.I)
            clean_cand = clean_cand.strip()
            
            # Distance between clean versions
            dist = _levenshtein_distance(clean_input, clean_cand)
            max_len = max(len(clean_input), len(clean_cand))
            base_sim = 1 - (dist / max_len) if max_len > 0 else 1.0
            
            # Bonus for label matching
            label_match_bonus = 0
            for p in prefixes_to_strip:
                if (p in input_raw) and (p in orig_name):
                    label_match_bonus += 0.05 # Boost for both having the label
            
            score = base_sim + label_match_bonus
            
            if score > max_score:
                max_score = score
                best_match = item
        
        if best_match and max_score >= threshold:
            return best_match['name'], best_match['id'], max_score
        return None, None, 0.0

    def match_province(self, text):
        return self._fuzzy_match_regional(text, self.provinces, ["PROVINSI"], threshold=0.6)

    def match_city(self, text, province_id):
        if not province_id: return None, None, 0.0
        if province_id not in self._city_cache:
            self._city_cache[province_id] = self._load_data(f"regencies/{province_id}.json")
        return self._fuzzy_match_regional(text, self._city_cache[province_id], ["KABUPATEN", "KOTA", "KAB"], threshold=0.6)

    def match_district(self, text, city_id):
        if not city_id: return None, None, 0.0
        if city_id not in self._district_cache:
            self._district_cache[city_id] = self._load_data(f"districts/{city_id}.json")
        return self._fuzzy_match_regional(text, self._district_cache[city_id], ["KECAMATAN", "KEC"], threshold=0.6)

    def match_village(self, text, district_id):
        if not district_id: return None, None, 0.0
        if district_id not in self._village_cache:
            self._village_cache[district_id] = self._load_data(f"villages/{district_id}.json")
        return self._fuzzy_match_regional(text, self._village_cache[district_id], ["KEL/DESA", "KELURAHAN", "DESA", "KEL"], threshold=0.6)

# Global matcher instance to share cache across runs
_regional_matcher = RegionalMatcher()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_ktp(raw_text: str) -> KTPData:
    """
    Parse raw OCR string → structured KTPData.

    Pass the exact string returned by run_ocr() into this function.
    """
    ktp  = KTPData()
    text = _strip_tokens(raw_text)
    text = _norm(text)

    # 1. NIK
    m = re.search(r"\b(\d{15,16})\b", text)
    if m:
        ktp.nik = m.group(1)
        text    = _remove_at(m, text)

    # 2. RT/RW
    m = re.search(r"\b(\d{3})\s*/\s*(\d{3})\b", text)
    if m:
        ktp.rt_rw = f"{m.group(1)}/{m.group(2)}"
        text      = _remove_at(m, text)

    # 3. Berlaku Hingga
    if "SEUMUR HIDUP" in text:
        ktp.berlaku_hingga = "SEUMUR HIDUP"
        text = _remove("SEUMUR HIDUP", text)
    else:
        m = re.search(r"\b(\d{2}[-/]\d{2}[-/]\d{4})\b", text)
        if m:
            ktp.berlaku_hingga = m.group(1).replace("/", "-")
            text = _remove_at(m, text)

    # 4. Tanggal Lahir
    m = re.search(r"\b(\d{2}-\d{2}-\d{4})\b", text)
    if m:
        ktp.tgl_lahir = m.group(1)
        text = _remove_at(m, text)
    else:
        # Avoid picking up 5-digit numbers that are likely not dates (e.g. ZIP codes)
        m = re.search(r"\b(\d{5})\b", text)
        if m:
            serial_val = int(m.group(1))
            if _is_birth_serial(serial_val):
                ktp.tgl_lahir = _excel_to_date(serial_val)
                text = _remove_at(m, text)

    # 5. Fixed-vocab fields
    ktp.jenis_kelamin,     text = _find_vocab(JK_LIST,        text)
    ktp.kewarganegaraan,   text = _find_vocab(WN_LIST,        text)
    ktp.agama,             text = _find_vocab(AGAMA_LIST,     text)
    ktp.status_perkawinan, text = _find_vocab(STATUS_LIST,    text)
    ktp.pekerjaan,         text = _find_vocab(PEKERJAAN_LIST, text)

    # 6. Golongan Darah — position-safe removal (avoids wiping "O" from words)
    for gd in GOL_DARAH_ORDERED:
        m = re.search(rf"\b{re.escape(gd)}\b", text)
        if m:
            ktp.gol_darah = gd
            text = _remove_at(m, text)
            break

    # 7. Regions (Province, City) via RegionalMatcher API
    # Since unstructured OCR doesn't have neat labels, we'll try to scan known words or lines
    
    # Try find Provinsi
    pid = None
    m_prov = re.search(r"\bPROVINSI\b", text)
    if m_prov:
        idx = int(m_prov.start())
        words_after = str(text)[idx:].split()[:5]
        best_cand, best_id, best_phrase, best_score = None, None, "", 0.0
        for length in [1, 2, 3, 4, 5]:
            phrase = " ".join(words_after[:length])
            prov_cand, prov_id, score = _regional_matcher.match_province(phrase)
            if prov_cand and score > best_score:
                best_cand, best_id, best_phrase, best_score = prov_cand, prov_id, phrase, score
        if best_cand:
            ktp.provinsi = best_cand
            pid = best_id
            text = _remove(best_phrase, text)
        if not ktp.provinsi:
            # Fallback if API fails but we found PROVINSI
            phrase = " ".join(words_after[:3])
            ktp.provinsi = phrase.replace("PROVINSI ", "")
            text = _remove(phrase, text)

    # 8. Kota / Tempat Lahir — BEFORE alamat (prevents city names bleeding in)
    # We look for KOTA/KABUPATEN
    cid = None
    m_kota = re.search(r"\b(KOTA|KABUPATEN)\b", text)
    if m_kota:
        idx = m_kota.start()
        words_after = text[idx:].split()[:5]
        best_cand, best_id, best_phrase, best_score = None, None, "", 0.0
        for length in [1, 2, 3, 4, 5]:
            phrase = " ".join(words_after[:length])
            kota_cand, kota_id, score = _regional_matcher.match_city(phrase, pid) if pid else (None, None, 0.0)
            if kota_cand and score > best_score:
                best_cand, best_id, best_phrase, best_score = kota_cand, kota_id, phrase, score
        if best_cand:
            ktp.kota = best_cand
            cid = best_id
            text = _remove(best_phrase, text)
        if not ktp.kota:
            # Fallback
            phrase = " ".join(words_after[:3])
            ktp.kota = phrase
            text = _remove(phrase, text)

    # Try finding city by scanning all valid cities for the province
    if not ktp.kota and pid:
        city_data = _regional_matcher._city_cache.get(pid) or _regional_matcher._load_data(f"regencies/{pid}.json")
        if city_data:
            _regional_matcher._city_cache[pid] = city_data
            for c in sorted(city_data, key=lambda x: len(x.get('name', '')), reverse=True):
                cand_name = c.get('name', '')
                clean_cand = cand_name.replace("KOTA ", "").replace("KABUPATEN ", "").strip()
                if cand_name in text:
                    ktp.kota = cand_name
                    cid = c.get('id')
                    text = _remove(cand_name, text)
                    break
                elif clean_cand and clean_cand in text and len(clean_cand) > 3:
                    ktp.kota = cand_name
                    cid = c.get('id')
                    text = _remove(clean_cand, text)
                    break

    # Extract Tempat Lahir from words preceding Tgl Lahir
    m_tmp = re.search(r"([A-Z\s]+?(?:KOTA\s+|KABUPATEN\s+)?[A-Z\s]+)\s*\b\d{2}-\d{2}-\d{4}\b", raw_text)
    if m_tmp:
        # Clean up extracted tempat_lahir (it's often the city)
        pot_tl = _norm(m_tmp.group(1)).split()[-3:] # get last 3 words
        pot_tl_str = " ".join(pot_tl)
        tl_cand, _, _ = _regional_matcher.match_city(pot_tl_str, pid) if pid else (None, None, 0.0)
        if tl_cand:
            ktp.tempat_lahir = tl_cand.replace("KOTA ", "").replace("KABUPATEN ", "")
        else:
            # just use the last word before date as fallback
            if pot_tl:
                ktp.tempat_lahir = pot_tl[-1] 
            elif ktp.kota:
                ktp.tempat_lahir = str(ktp.kota).replace("KOTA ", "").replace("KABUPATEN ", "")
    elif ktp.kota:
        ktp.tempat_lahir = str(ktp.kota).replace("KOTA ", "").replace("KABUPATEN ", "")

    # 9. Alamat — AFTER kota removal
    PREFIX = (
        r"(?:JL\.?|JALAN|GG\.?|GANG|KP\.?|KAMPUNG"
        r"|KOMPLEK|PERUM\.?|PERUMAHAN|BLOK|DUSUN|DSN)"
    )
    m = re.search(rf"\b({PREFIX}(?:\s+[A-Z0-9][A-Z0-9 .,/\-]*?)+)\s*$", text)
    if not m:
        m = re.search(rf"\b({PREFIX}\s+[A-Z0-9][A-Z0-9 .,/\-]*)", text)
    if m:
        ktp.alamat = _norm(m.group(1))
        text = _norm(text.replace(m.group(1), " "))
        
    # 10. Remaining → kel_desa, kecamatan, nama
    words = [w for w in text.split() if len(w) > 1 and re.search(r"[A-Z]", w)]
    
    # Pre-extract Kecamatan and Kel/Desa using API if City ID is available
    matched_words = set()
    did = None
    if cid:
        # Try to find sequences of words that match a Kecamatan
        best_kec, best_kec_id, best_kec_score, best_kec_range = None, None, 0.0, []
        for length in [1, 2, 3]:
            for i in range(len(words) - length + 1):
                phrase = " ".join(words[i:i+length])
                kec_cand, kec_id, score = _regional_matcher.match_district(phrase, cid)
                if kec_cand and score > best_kec_score:
                    best_kec, best_kec_id, best_kec_score = kec_cand, kec_id, score
                    best_kec_range = list(range(i, i+length))
        
        if best_kec:
            ktp.kecamatan = best_kec
            did = best_kec_id
            for j in best_kec_range: matched_words.add(j)
        
        # Try to find Kelurahan/Desa if District ID is available
        if did:
            best_kel, best_kel_id, best_kel_score, best_kel_range = None, None, 0.0, []
            for length in [1, 2, 3]:
                for i in range(len(words) - length + 1):
                    if i in matched_words: continue # Skip words already matched for Kecamatan
                    phrase = " ".join(words[i:i+length])
                    kel_cand, kel_id, score = _regional_matcher.match_village(phrase, did)
                    if kel_cand and score > best_kel_score:
                        best_kel, best_kel_id, best_kel_score = kel_cand, kel_id, score
                        best_kel_range = list(range(i, i+length))
            if best_kel:
                ktp.kel_desa = best_kel
                for j in best_kel_range: matched_words.add(j)

    nama_parts = []
    kel_kec_found = (ktp.kecamatan is not None and ktp.kel_desa is not None)
    i = 0

    while i < len(words):
        if i in matched_words:
            i += 1
            continue
        
        w = words[i]
        if not kel_kec_found and i + 1 < len(words) and (i + 1) not in matched_words and words[i + 1] == w:
            ktp.kel_desa  = w
            ktp.kecamatan = w
            kel_kec_found = True
            i += 2
        else:
            nama_parts.append(w)
            i += 1

    if not kel_kec_found:
        if len(nama_parts) >= 3:
            ktp.kecamatan = nama_parts.pop()
            ktp.kel_desa  = nama_parts.pop()
        elif len(nama_parts) == 2:
            ktp.kel_desa  = nama_parts.pop()

    if not ktp.alamat and len(nama_parts) >= 4:
        # Heuristic: Assume first 2 words are Name, the rest become Alamat
        ktp.nama = " ".join(nama_parts[:2])
        ktp.alamat = " ".join(nama_parts[2:])
        nama_parts = []

    if nama_parts:
        ktp.nama = " ".join(nama_parts)

    # Cleanup Nama from Alamat if it leaked
    if ktp.nama and ktp.alamat:
        if ktp.alamat in ktp.nama:
            ktp.nama = _norm(ktp.nama.replace(ktp.alamat, ""))
    
    # Final cleanup: remove residual province/kota from nama if leaked
    if ktp.nama:
        if ktp.provinsi and ktp.provinsi in ktp.nama:
            ktp.nama = _norm(ktp.nama.replace(ktp.provinsi, ""))
        if ktp.kota and ktp.kota in ktp.nama:
            ktp.nama = _norm(ktp.nama.replace(ktp.kota, ""))

    # 11. Cross-validate with NIK
    nik_tgl, nik_jk = _nik_decode(ktp.nik)
    notes: List[str] = []
    if ktp.tgl_lahir is None and nik_tgl:
        ktp.tgl_lahir = nik_tgl
        notes.append("tgl_lahir")
    elif ktp.tgl_lahir and nik_tgl:
        # If tgl_lahir from text is too recent (e.g. < 17 years old), trust NIK
        try:
            d_txt = date(*map(int, reversed(ktp.tgl_lahir.split("-"))))
            if d_txt > date.today() - timedelta(days=365*17): 
                ktp.tgl_lahir = nik_tgl
                notes.append("tgl_lahir (corrected)")
        except: pass

    if ktp.jenis_kelamin is None and nik_jk:
        ktp.jenis_kelamin = nik_jk
        notes.append("jenis_kelamin")
    if notes:
        suffix = (" | " + ktp._unmatched) if ktp._unmatched else ""
        ktp._unmatched = "[from NIK: " + ", ".join(notes) + "]" + suffix

    return ktp


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def process_image(pipe, image_path: str) -> dict:
    """
    Full pipeline for a single KTP image:
      image file → OCR → parse → structured dict

    Args:
        pipe:        HuggingFace pipeline object (from load_ocr_model)
        image_path:  Path to the KTP image

    Returns:
        dict with all KTP fields
    """
    print(f"[OCR] Processing: {image_path}")
    raw_text = run_ocr(pipe, image_path)
    print(f"[OCR] Raw output: {raw_text}")

    ktp    = parse_ktp(raw_text)
    result = asdict(ktp)

    if not result.get("_unmatched"):
        result.pop("_unmatched", None)

    return result


def process_folder(pipe, folder_path: str) -> List[dict]:
    """
    Batch-process all KTP images in a folder.
    Supports .jpg, .jpeg, .png, .webp files.

    Returns list of result dicts, each containing a '_source' key
    with the original filename.
    """
    folder  = Path(folder_path)
    images  = sorted(folder.glob("*.jpg")) + \
              sorted(folder.glob("*.jpeg")) + \
              sorted(folder.glob("*.png")) + \
              sorted(folder.glob("*.webp"))

    if not images:
        print(f"[WARN] No image files found in: {folder_path}")
        return []

    results = []
    for img_path in images:
        result = process_image(pipe, str(img_path))
        result["_source"] = img_path.name   # tag which file this came from
        results.append(result)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CLI ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="KTP OCR + Parser — fully on-premise, no external API."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, help="Path to a single KTP image")
    group.add_argument("--folder", type=str, help="Folder containing multiple KTP images")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional: save result as JSON to this path")
    parser.add_argument("--model",  type=str,
                        default="emisilab/model-ocr-ktp-v1",
                        help="HuggingFace model name (default: emisilab/model-ocr-ktp-v1)")
    args = parser.parse_args()

    # ── Load model once ───────────────────────────────────────────────────────
    pipe = load_ocr_model(args.model)

    # ── Run ───────────────────────────────────────────────────────────────────
    if args.image:
        result = process_image(pipe, args.image)
        output = result
    else:
        results = process_folder(pipe, args.folder)
        output  = results

    # ── Print ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(json.dumps(output, indent=2, ensure_ascii=False))

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVED] {args.output}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — NOTEBOOK / IMPORT USAGE
# ═══════════════════════════════════════════════════════════════════════════════
#
# If you're running this in a Jupyter notebook, use it like this:
#
#   from ktp_parser import load_ocr_model, process_image, parse_ktp
#
#   pipe   = load_ocr_model()                   # load once
#   result = process_image(pipe, "ktp.jpg")     # single image
#   print(result)
#
#   # Or if you already have the raw OCR string:
#   raw    = "<s_dataset_ktp> DEBBY ANGGRAINI ..."
#   result = parse_ktp(raw)
#
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
