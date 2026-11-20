import asyncio

from app.llm import MockLLM
from app.memory import Conversation, TokenBudget, cap_tokens
from app.tokens import count_text_tokens


def ask(llm, messages):
    return asyncio.run(llm.chat(messages)).content


def test_llm_is_stateless_until_history_is_sent():
    llm = MockLLM()
    ask(llm, [{"role": "user", "content": "My name is Priya."}])
    assert "don't have" in ask(llm, [{"role": "user", "content": "What is my name?"}])
    full = [{"role": "user", "content": "My name is Priya."},
            {"role": "assistant", "content": "Nice to meet you."},
            {"role": "user", "content": "What is my name?"}]
    assert "Priya" in ask(llm, full)


def test_conversation_stores_and_sends_full_history():
    conv = Conversation("sys")
    conv.add_user("hello")
    conv.add_assistant("hi")
    conv.add_user("again")
    sent = conv.build_messages()
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert conv.summary == ""                       # nothing folded yet


def test_sliding_window_folds_old_turns_but_keeps_early_fact():
    llm = MockLLM()
    conv = Conversation("sys", window_turns=3)
    conv.add_user("My name is Priya and my order number is 1042.")
    conv.add_assistant("Noted.")
    for i in range(10):
        conv.add_user(f"Tell me about topic {i} in detail.")
        conv.add_assistant("Sure.")
    conv.add_user("What was my order number?")
    sent = conv.build_messages()
    assert sum(1 for m in conv.active if m["role"] == "user") <= 3
    assert "1042" in conv.summary
    assert "1042" in asyncio.run(llm.chat(sent)).content
    assert len(conv.messages) == 23                  # the FULL log is never lost


def test_token_budget_keeps_history_under_3000():
    conv = Conversation("sys", window_turns=100)     # window disabled -> only the budget folds
    for _ in range(20):
        conv.add_user("word " * 400)
        conv.add_assistant("reply " * 100)
    conv.add_user("last question")
    conv.build_messages()
    assert conv.history_tokens() <= TokenBudget().history
    stats = conv.stats()
    assert stats["full_history_tokens"] > stats["sent_history_tokens"]


def test_system_prompt_is_capped_to_500_tokens():
    conv = Conversation("x " * 5000)
    conv.add_user("hi")
    system = conv.build_messages()[0]["content"]
    assert count_text_tokens(system) <= 500


def test_pop_last_and_cap_tokens():
    conv = Conversation("sys")
    conv.add_user("oops")
    conv.pop_last()
    assert conv.messages == []
    assert cap_tokens("short", 100) == "short"
