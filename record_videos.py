"""
Grava vídeos do agente jogando Boxing conforme o treino avança.

Fica observando modelos/checkpoints/ e, para cada checkpoint novo, grava uma luta
completa em resultados/videos/. Pode rodar ao mesmo tempo que o train_boxing.py.

Uso:
    .venv\\Scripts\\python record_videos.py            # observa até o treino terminar
    .venv\\Scripts\\python record_videos.py --once     # só grava o que já existe e sai
"""

import argparse
import glob
import os
import re
import time

import ale_py
import gymnasium as gym
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

ENV_ID = "ALE/Boxing-v5"
MODEL_DIR = "modelos"
CKPT_DIR = os.path.join(MODEL_DIR, "checkpoints")
VIDEO_DIR = os.path.join("resultados", "videos")
SCALE = 3          # aumenta a imagem do Atari (210x160) para ficar visível
BAR = 42           # faixa de texto no topo (630 + 42 = 672, múltiplo de 16 para o codec)
SEED = 42          # mesma semente em todos os vídeos, para comparar


def get_font(size: int = 24):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow antigo
        return ImageFont.load_default()


def add_label(frame: np.ndarray, text: str, font, scale: int = SCALE, bar: int = BAR) -> np.ndarray:
    big = frame.repeat(scale, axis=0).repeat(scale, axis=1)
    img = Image.new("RGB", (big.shape[1], big.shape[0] + bar), (20, 20, 20))
    img.paste(Image.fromarray(big), (0, bar))
    ImageDraw.Draw(img).text((12, bar // 5), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def play(model_path, max_steps: int = 2000, seed: int = SEED):
    """Joga uma luta e devolve (frames coloridos 210x160, placar final).

    model_path=None joga com ações aleatórias (linha de base sem treino).
    """
    env = make_atari_env(
        ENV_ID, n_envs=1, seed=seed,
        env_kwargs={"frameskip": 1, "render_mode": "rgb_array"},
    )
    env = VecFrameStack(env, n_stack=4)
    # CPU para não disputar a GPU com um treino rodando ao mesmo tempo.
    model = PPO.load(model_path, device="cpu") if model_path else None

    obs = env.reset()
    frames, score = [], None
    for _ in range(max_steps):  # uma luta dura ~1800 passos (2 minutos de jogo)
        frames.append(env.get_images()[0])
        if model is None:
            action = [env.action_space.sample()]
        else:
            action, _ = model.predict(obs, deterministic=False)
        obs, _, done, info = env.step(action)
        if done[0]:
            # Placar real (o Monitor guarda a recompensa sem o "clip" do AtariWrapper).
            score = info[0]["episode"]["r"]
            break
    env.close()
    return frames, score


def record(model_path, out_path: str, label: str) -> float:
    frames, score = play(model_path)
    font = get_font()
    frames = [add_label(f, label, font) for f in frames]
    # 15 fps = velocidade real do jogo (60 fps / 4 frames por ação).
    imageio.mimsave(out_path, frames, fps=15, macro_block_size=1)
    return score


def steps_of(path: str) -> int:
    m = re.search(r"_(\d+)_steps", path)
    return int(m.group(1)) if m else -1


def fmt_steps(n: int) -> str:
    return f"{n / 1_000_000:.1f}M" if n >= 1_000_000 else f"{n // 1000}k"


def process(path, name: str, label: str):
    out = os.path.join(VIDEO_DIR, name)
    if os.path.exists(out):
        return
    print(f"Gravando {out} ...", flush=True)
    score = record(path, out, label)
    print(f"  pronto: placar {score:+.0f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=30, help="segundos entre verificações")
    args = parser.parse_args()
    os.makedirs(VIDEO_DIR, exist_ok=True)

    process(None, "00_agente_aleatorio.mp4", "Agente aleatorio (sem treino)")

    while True:
        for ckpt in sorted(glob.glob(os.path.join(CKPT_DIR, "*.zip")), key=steps_of):
            # Ignora arquivos que ainda podem estar sendo escritos.
            if time.time() - os.path.getmtime(ckpt) < 10:
                continue
            n = steps_of(ckpt)
            process(ckpt, f"boxing_{n:09d}_passos.mp4", f"Treino: {fmt_steps(n)} passos")

        final = os.path.join(MODEL_DIR, "boxing_ppo_final.zip")
        if args.once or (os.path.exists(final) and time.time() - os.path.getmtime(final) > 10):
            if os.path.exists(final):
                process(final, "modelo_final.mp4", "Modelo final")
            best = os.path.join(MODEL_DIR, "best_model.zip")
            if os.path.exists(best):
                process(best, "melhor_modelo.mp4", "Melhor modelo (avaliacao)")
            break
        time.sleep(args.interval)

    print("Fim.", flush=True)


if __name__ == "__main__":
    main()
