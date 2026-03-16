from ktp_parser import parse_ktp
import json

raw_text = "<s_dataset_ktp> BELUM KAWIN 015/003 PROVINSI DKI JAKARTA PELAJAR/MAHASISWA 3175032708020008 ERLAN EFFENDY WNI CIPINANG MUARA JATINEGARA JAKARTA TIMUR LAKI-LAKI 44865 JAKARTA TIMUR SEUMUR HIDUP CIPINANG MUARA II KATHOLIK"

result = parse_ktp(raw_text)
print(json.dumps(result.__dict__, indent=2))
