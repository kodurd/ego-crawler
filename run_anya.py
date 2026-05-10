#!/usr/bin/env python3
"""Точка входа: агент Аня Соколова — московская отличница с комплексом перфекциониста.

Запуск:
    # Вписать DASHSCOPE_API_KEY в .env, затем:
    py run_anya.py

    # Headless=false — видимый браузер:
    PLAYWRIGHT_HEADLESS=false py run_anya.py
"""
from __future__ import annotations

import os
import sys
import signal
from pathlib import Path

# Загружаем .env до любых импортов, читающих переменные окружения
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from ego_crawler.domain.entities.session import Session
from ego_crawler.domain.value_objects.budget import Budget
from ego_crawler.domain.personas.anya_sokolova import create_anya_sokolova
from ego_crawler.infrastructure.qwen.qwen_llm_client import QwenLLMClient
from ego_crawler.infrastructure.playwright.playwright_web_tools import PlaywrightWebTools
from ego_crawler.infrastructure.persistence.duckdb.duckdb_session_repository import DuckDBSessionRepository
from ego_crawler.infrastructure.persistence.duckdb.duckdb_step_repository import DuckDBStepRepository
from ego_crawler.application.use_cases.execute_agent_step import ExecuteAgentStep, StepResult
from ego_crawler.application.services.prompt_builder import PromptBuilder

# ── ANSI цвета ────────────────────────────────────────────────────────────────

_NO_COLOR = not sys.stdout.isatty() or os.environ.get("NO_COLOR")

def _c(code: str, text: str) -> str:
    return text if _NO_COLOR else f"\033[{code}m{text}\033[0m"

def cyan(t):    return _c("96", t)
def yellow(t):  return _c("93", t)
def green(t):   return _c("92", t)
def red(t):     return _c("91", t)
def dim(t):     return _c("2",  t)
def bold(t):    return _c("1",  t)
def blue(t):    return _c("94", t)


# ── Консольный вывод ──────────────────────────────────────────────────────────

_W = 66  # ширина блока

def _line(char: str = "─") -> str:
    return dim(char * _W)

def _header(title: str) -> str:
    pad = _W - 2 - len(title)
    return cyan("╔" + "═" * (_W - 2) + "╗") + "\n" + \
           cyan("║ ") + bold(title) + " " * pad + cyan("║") + "\n" + \
           cyan("╚" + "═" * (_W - 2) + "╝")

def print_session_header(session: Session, model: str, budget_min: int, max_steps: int) -> None:
    p = session.persona
    title = f"EGO-CRAWLER  ·  {p.name}  ·  {p.archetype}, {p.age} лет"
    print()
    print(_header(title))
    print()
    print(f"  Модель:  {yellow(model)}")
    print(f"  Бюджет:  {budget_min} мин  ·  Макс. шагов: {max_steps}")
    print(f"  Сессия:  {dim(str(session.id)[:12] + '...')}")
    print()


def print_step_header(n: int, session: Session) -> None:
    td = session.budget.as_timedelta()
    mood = session.current_emotion.mood.name
    intensity = session.current_emotion.intensity
    print(_line())
    print(
        f"  {bold(f'ШАГ {n}')}  ·  "
        f"{yellow(mood)} ({intensity:.2f})  ·  "
        f"{dim(str(td))} осталось"
    )
    print(_line())
    print()


def print_step_result(result: StepResult) -> None:
    thought = result.thought

    # Мысль
    thought_text = (thought.thought_content or "").strip()
    print(f"  {blue('[МЫСЛЬ]')}  {thought_text}")
    print()

    # Действие + наблюдение
    if result.action:
        import json
        params_str = json.dumps(result.action.action_params or {}, ensure_ascii=False)
        print(f"  {yellow('[ДЕЙСТВИЕ]')}  {bold(result.action.action_tool)}")
        print(f"  {dim(params_str)}")
        print()

        if result.observation:
            raw = (result.observation.observation_raw or "").strip()
            summary = result.observation.observation_summary
            if summary:
                status = green("✓ успех") if not summary else red(f"✗ {summary[:80]}")
            else:
                status = green("✓ успех")
            preview = raw[:120] + ("…" if len(raw) > 120 else "")
            print(f"  {green('[РЕЗУЛЬТАТ]')}  {status}")
            if preview:
                print(f"  {dim(preview)}")
            print()

    # Статистика шага
    lat = thought.latency_ms or 0
    tok_in  = thought.token_count_input  or 0
    tok_out = thought.token_count_output or 0
    print(f"  {dim(f'⏱  {lat} ms  ·  in: {tok_in}  out: {tok_out} токенов')}")
    print()


def print_summary(session: Session, step_repo) -> None:
    steps = step_repo.get_by_session(session.id)
    thoughts = [s for s in steps if s.step_type.name == "THOUGHT"]
    actions  = [s for s in steps if s.step_type.name == "ACTION"]

    total_in  = sum(s.token_count_input  or 0 for s in thoughts)
    total_out = sum(s.token_count_output or 0 for s in thoughts)
    # Примерная стоимость qwen-plus: ~$0.0004/1K вх + $0.0012/1K вых
    approx_usd = (total_in / 1000 * 0.0004) + (total_out / 1000 * 0.0012)

    budget_used = session.budget.total_minutes * 60 - session.budget.remaining_seconds
    budget_used_min = budget_used // 60
    budget_used_sec = budget_used % 60

    print()
    print(cyan("═" * _W))
    print(f"  {bold('ИТОГ СЕССИИ')}")
    print(cyan("═" * _W))
    print(f"  Шагов выполнено:     {bold(str(len(thoughts)))}")
    print(f"  Действий:            {len(actions)}")
    print(f"  Израсходовано:       {budget_used_min}:{budget_used_sec:02d} / "
          f"{session.budget.total_minutes}:00")
    print(f"  Финальное настроение: "
          f"{yellow(session.current_emotion.mood.name)} "
          f"({session.current_emotion.intensity:.2f})")
    print(f"  Токены вход/выход:   {total_in} / {total_out}")
    print(f"  Примерная стоимость: {dim(f'~${approx_usd:.4f}')}")
    print(cyan("═" * _W))
    print()

# ── Конфигурация из .env ──────────────────────────────────────────────────────

def _cfg_str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()

def _cfg_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default

def _cfg_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, str(default)).strip().lower()
    return val not in ("false", "0", "no", "off")

# ── Главная функция ───────────────────────────────────────────────────────────

def main() -> None:
    model        = _cfg_str ("QWEN_MODEL",            "qwen-plus")
    budget_min   = _cfg_int ("AGENT_BUDGET_MINUTES",  30)
    max_steps    = _cfg_int ("AGENT_MAX_STEPS",        15)
    temperature  = float(_cfg_str("AGENT_TEMPERATURE", "0.75"))
    headless     = _cfg_bool("PLAYWRIGHT_HEADLESS",    True)
    timeout_ms   = _cfg_int ("PLAYWRIGHT_TIMEOUT_MS",  30_000)
    db_path      = _cfg_str ("DB_PATH",               "ego_crawler.duckdb")

    # Инициализация — валидация API ключа происходит здесь
    try:
        llm = QwenLLMClient(model=model)
    except ValueError as exc:
        print(red(f"\n  Ошибка: {exc}"))
        print(dim("  Добавьте DASHSCOPE_API_KEY в файл .env\n"))
        sys.exit(1)

    anya           = create_anya_sokolova()
    session_repo   = DuckDBSessionRepository(db_path)
    step_repo      = DuckDBStepRepository(db_path)
    prompt_builder = PromptBuilder()

    # Создаём сессию напрямую с полной персоной (обходим CreateSession,
    # которая пересоздаёт Persona из примитивов и теряет system_prompt)
    anya = anya.with_context(model=model)
    session = Session(
        persona=anya,
        budget=Budget.from_minutes(budget_min),
    )
    session_repo.save(session)

    # Обработчик Ctrl+C — напечатает итог перед выходом
    _interrupted = [False]
    def _handle_sigint(sig, frame):
        print(yellow("\n\n  Прервано пользователем."))
        _interrupted[0] = True
    signal.signal(signal.SIGINT, _handle_sigint)

    print_session_header(session, model, budget_min, max_steps)

    step_uc = ExecuteAgentStep(
        session_repo   = session_repo,
        step_repo      = step_repo,
        llm_client     = llm,
        web_tools      = None,  # заменяется внутри with-блока
        prompt_builder = prompt_builder,
    )

    with PlaywrightWebTools(headless=headless, timeout=timeout_ms) as browser:
        step_uc.web_tools = browser  # подключаем реальный браузер

        step_n = 0
        while session.is_active() and step_n < max_steps and not _interrupted[0]:
            step_n += 1
            print_step_header(step_n, session)

            result  = step_uc.execute(session.id)
            session = result.session
            session_repo.save(session)

            print_step_result(result)

    print_summary(session, step_repo)


if __name__ == "__main__":
    main()
