"""
Treina um agente PPO para jogar Atari Boxing (ALE/Boxing-v5).

Uso:
    .venv\\Scripts\\python train_boxing.py                      # treino completo (10M passos)
    .venv\\Scripts\\python train_boxing.py --timesteps 100000   # teste rápido
    .venv\\Scripts\\python train_boxing.py --resume modelos/checkpoints/boxing_ppo_5000000_steps.zip

Saídas:
    modelos/best_model.zip       -> melhor modelo segundo as avaliações periódicas
    modelos/boxing_ppo_final.zip -> modelo ao fim do treino
    modelos/checkpoints/         -> cópias salvas a cada --checkpoint-freq passos
    resultados/tensorboard/      -> métricas para o TensorBoard
"""

import argparse
import os

import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage

gym.register_envs(ale_py)

ENV_ID = "ALE/Boxing-v5"
MODEL_DIR = "modelos"
TB_DIR = "./resultados/tensorboard/"

# O ALE v5 já pula 4 frames por padrão, e o AtariWrapper do SB3 pula mais 4.
# Com frameskip=1 aqui, só o AtariWrapper pula frames (total de 4).
ENV_KWARGS = {"frameskip": 1}


def make_env(n_envs: int, seed: int, subproc: bool = False):
    # make_atari_env aplica Monitor + AtariWrapper: escala de cinza, 84x84,
    # pula 4 frames, NOOPs aleatórios no reset e recompensa limitada a {-1, 0, +1}.
    env = make_atari_env(
        ENV_ID,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=ENV_KWARGS,
        vec_env_cls=SubprocVecEnv if subproc else None,
    )
    # Empilha os últimos 4 frames para a rede perceber o movimento
    # (para onde o oponente está indo, se ele está no meio de um soco...).
    env = VecFrameStack(env, n_stack=4)
    return VecTransposeImage(env)


def linear_schedule(initial: float):
    # Diminui o valor linearmente até 0 ao longo do treino
    # (progress_remaining vai de 1 até 0).
    return lambda progress_remaining: progress_remaining * initial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-freq", type=int, default=250_000, help="avaliar a cada N passos (total)")
    parser.add_argument("--checkpoint-freq", type=int, default=1_000_000, help="salvar a cada N passos (total)")
    parser.add_argument("--resume", type=str, default=None, help="caminho de um .zip para continuar o treino")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    # N_ENVS cópias do jogo em processos separados (aproveita os vários núcleos da CPU).
    env = make_env(args.n_envs, args.seed, subproc=True)
    eval_env = make_env(1, args.seed + 1000)

    if args.resume:
        model = PPO.load(args.resume, env=env, tensorboard_log=TB_DIR)
        print(f"Continuando treino de {args.resume} ({model.num_timesteps} passos já feitos)")
    else:
        # Hiperparâmetros do RL Baselines3 Zoo para PPO em Atari.
        model = PPO(
            "CnnPolicy",
            env,
            # N_STEPS: passos coletados POR AMBIENTE antes de cada atualização.
            # Rollout total = n_steps * n_envs = 128 * 8 = 1024 transições.
            n_steps=128,
            # BATCH_SIZE: amostras em cada minibatch da otimização da rede.
            batch_size=256,
            # N_EPOCHS: quantas vezes o PPO percorre o mesmo rollout.
            n_epochs=4,
            # LEARNING_RATE: começa em 2.5e-4 e cai linearmente até 0.
            learning_rate=linear_schedule(2.5e-4),
            # CLIP_RANGE: limita o quanto a política muda por atualização (o "clip" do PPO).
            clip_range=linear_schedule(0.1),
            # ENT_COEF: bônus de entropia, incentiva a exploração
            # (testar socos e direções diferentes).
            ent_coef=0.01,
            vf_coef=0.5,
            gamma=0.99,
            gae_lambda=0.95,
            seed=args.seed,
            verbose=1,
            tensorboard_log=TB_DIR,
        )

    print("Dispositivo:", model.device)

    # As frequências dos callbacks contam chamadas de env.step(); cada chamada
    # avança n_envs passos, por isso a divisão.
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            save_path=os.path.join(MODEL_DIR, "checkpoints"),
            name_prefix="boxing_ppo",
        ),
        # Joga lutas completas de tempos em tempos e salva o melhor modelo.
        EvalCallback(
            eval_env,
            n_eval_episodes=5,
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            best_model_save_path=MODEL_DIR,
            log_path=MODEL_DIR,
            deterministic=False,
        ),
    ])

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name="PPO_boxing",
            reset_num_timesteps=not args.resume,
            progress_bar=True,
        )
    finally:
        # Salva mesmo se o treino for interrompido (Ctrl+C).
        model.save(os.path.join(MODEL_DIR, "boxing_ppo_final"))
        print("Modelo salvo em", os.path.join(MODEL_DIR, "boxing_ppo_final.zip"))
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
