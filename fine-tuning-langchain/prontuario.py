from pathlib import Path
import pandas as pd

BASE_PATH = Path(__file__).parent / "data" / "base_mock_prontuarios.xlsx"


def buscar_prontuario_por_cpf(cpf: str) -> dict:
    atendimentos = pd.read_excel(BASE_PATH, sheet_name="Atendimentos Gerais")
    internacoes = pd.read_excel(BASE_PATH, sheet_name="Internacoes")
    alergias = pd.read_excel(BASE_PATH, sheet_name="Alergias")

    return {
        "atendimentos": atendimentos[atendimentos["CPF"] == cpf].to_dict("records"),
        "internacoes": internacoes[internacoes["CPF"] == cpf].to_dict("records"),
        "alergias": alergias[alergias["CPF"] == cpf].to_dict("records"),
    }

def cpf_existe_no_prontuario(cpf: str) -> bool:
    if not cpf.strip():
        return False

    atendimentos = pd.read_excel(BASE_PATH, sheet_name="Atendimentos Gerais")
    internacoes = pd.read_excel(BASE_PATH, sheet_name="Internacoes")
    alergias = pd.read_excel(BASE_PATH, sheet_name="Alergias")

    existe_em_atendimentos = atendimentos["CPF"].astype(str).eq(cpf).any()
    existe_em_internacoes = internacoes["CPF"].astype(str).eq(cpf).any()
    existe_em_alergias = alergias["CPF"].astype(str).eq(cpf).any()

    return existe_em_atendimentos or existe_em_internacoes or existe_em_alergias

def montar_contexto_prontuario(prontuario: dict) -> str:
    partes = []

    if prontuario["alergias"]:
        alergias = ", ".join(item["Alergia"] for item in prontuario["alergias"])
        partes.append(f"Alergias registradas: {alergias}.")

    if prontuario["atendimentos"]:
        partes.append("Atendimentos recentes:")
        for item in prontuario["atendimentos"][-5:]:
            partes.append(
                f"- {item['DataVisita']}: {item['Motivo da visita']} | Resultado: {item['Resultado']}"
            )

    if prontuario["internacoes"]:
        partes.append("Internações:")
        for item in prontuario["internacoes"][-5:]:
            partes.append(
                f"- Entrada: {item['DataEntrada']} | Saída: {item['DataSaida']} | Motivo: {item['Motivo']}"
            )

    return "\n".join(partes) or "Nenhum registro encontrado para este paciente."

