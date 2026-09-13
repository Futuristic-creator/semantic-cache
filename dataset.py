"""
Benchmark dataset for the semantic caching comparison.

Design: a small set of CANONICAL queries represent prompts that have
already been answered once and cached. TEST_QUERIES represent new
incoming traffic. Each test query is labeled with the canonical id it
SHOULD match (or None if it should NOT hit the cache) and a category:

  - exact       : identical text to a canonical query (easy positive)
  - paraphrase  : semantically equivalent, different wording (real positive)
  - trap        : high lexical/semantic overlap but DIFFERENT intent
                   (the classic semantic-cache failure mode -- should NOT
                   hit, but naive cosine-similarity systems often do)
  - unrelated   : genuinely different topic (easy negative)

This is a hand-curated "AI customer-support app" style dataset rather
than a scraped one, since the sandbox has no internet access to pull
QQP/STS-B. It's small by design -- built for a clear demo/ADR, not
statistical publication.
"""

CANONICAL_QUERIES = [
    {"id": "c1", "topic": "password_reset", "text": "How do I reset my password?"},
    {"id": "c2", "topic": "cancel_subscription", "text": "How do I cancel my subscription?"},
    {"id": "c3", "topic": "refund", "text": "How do I request a refund for my last order?"},
    {"id": "c4", "topic": "track_order", "text": "Where is my order? I want to track it."},
    {"id": "c5", "topic": "two_factor", "text": "How do I enable two-factor authentication on my account?"},
    {"id": "c6", "topic": "change_email", "text": "How do I change the email address on my account?"},
    {"id": "c7", "topic": "delete_account", "text": "How do I permanently delete my account?"},
    {"id": "c8", "topic": "payment_failed", "text": "Why did my payment fail at checkout?"},
    {"id": "c9", "topic": "update_card", "text": "How do I update my saved credit card details?"},
    {"id": "c10", "topic": "invoice", "text": "Where can I download my invoice for last month?"},
]

TEST_QUERIES = [
    # ---- password_reset (c1) ----
    {"text": "How do I reset my password?", "match": "c1", "category": "exact"},
    {"text": "I forgot my password, how do I reset it?", "match": "c1", "category": "paraphrase"},
    {"text": "Can you help me set a new password, I'm locked out", "match": "c1", "category": "paraphrase"},
    {"text": "password reset steps please", "match": "c1", "category": "paraphrase"},
    {"text": "How do I reset my password expiry policy for the whole team?", "match": None, "category": "trap"},
    {"text": "How do I reset my password manager app, not my account", "match": None, "category": "trap"},

    # ---- cancel_subscription (c2) ----
    {"text": "How do I cancel my subscription?", "match": "c2", "category": "exact"},
    {"text": "I want to stop my subscription, what's the process?", "match": "c2", "category": "paraphrase"},
    {"text": "please cancel my plan, I don't want to renew", "match": "c2", "category": "paraphrase"},
    {"text": "how to unsubscribe from the service", "match": "c2", "category": "paraphrase"},
    {"text": "How do I cancel my subscription renewal reminder emails but keep the subscription?", "match": None, "category": "trap"},
    {"text": "How do I cancel a subscription gift I sent to a friend?", "match": None, "category": "trap"},

    # ---- refund (c3) ----
    {"text": "How do I request a refund for my last order?", "match": "c3", "category": "exact"},
    {"text": "I want my money back for the order I placed", "match": "c3", "category": "paraphrase"},
    {"text": "can I get a refund on my recent purchase", "match": "c3", "category": "paraphrase"},
    {"text": "refund process for an order please", "match": "c3", "category": "paraphrase"},
    {"text": "How do I request a refund policy document for my accountant?", "match": None, "category": "trap"},
    {"text": "How do I request a replacement, not a refund, for my last order?", "match": None, "category": "trap"},

    # ---- track_order (c4) ----
    {"text": "Where is my order? I want to track it.", "match": "c4", "category": "exact"},
    {"text": "can you tell me the status of my delivery", "match": "c4", "category": "paraphrase"},
    {"text": "my package hasn't arrived, where is it", "match": "c4", "category": "paraphrase"},
    {"text": "order tracking info please", "match": "c4", "category": "paraphrase"},
    {"text": "Where is my order confirmation email, I never got it", "match": None, "category": "trap"},
    {"text": "Where is my order history so I can reorder the same thing", "match": None, "category": "trap"},

    # ---- two_factor (c5) ----
    {"text": "How do I enable two-factor authentication on my account?", "match": "c5", "category": "exact"},
    {"text": "how do I turn on 2FA for my login", "match": "c5", "category": "paraphrase"},
    {"text": "I want to add extra security with a code when I log in", "match": "c5", "category": "paraphrase"},
    {"text": "set up two factor auth please", "match": "c5", "category": "paraphrase"},
    {"text": "How do I enable two-factor authentication on my old account that I no longer have access to?", "match": None, "category": "trap"},
    {"text": "How do I disable two-factor authentication, it's locking me out?", "match": None, "category": "trap"},

    # ---- change_email (c6) ----
    {"text": "How do I change the email address on my account?", "match": "c6", "category": "exact"},
    {"text": "I need to update my account email to a new one", "match": "c6", "category": "paraphrase"},
    {"text": "how can I switch the email linked to my profile", "match": "c6", "category": "paraphrase"},
    {"text": "change my login email please", "match": "c6", "category": "paraphrase"},
    {"text": "How do I change the email notification settings on my account?", "match": None, "category": "trap"},
    {"text": "How do I change the email address of my company's billing contact, not mine?", "match": None, "category": "trap"},

    # ---- delete_account (c7) ----
    {"text": "How do I permanently delete my account?", "match": "c7", "category": "exact"},
    {"text": "I want to close my account for good", "match": "c7", "category": "paraphrase"},
    {"text": "how do I remove my profile completely", "match": "c7", "category": "paraphrase"},
    {"text": "delete my account please, permanently", "match": "c7", "category": "paraphrase"},
    {"text": "How do I permanently delete a single item from my order history, not my account?", "match": None, "category": "trap"},
    {"text": "How do I temporarily deactivate, not permanently delete, my account?", "match": None, "category": "trap"},

    # ---- payment_failed (c8) ----
    {"text": "Why did my payment fail at checkout?", "match": "c8", "category": "exact"},
    {"text": "my card was declined when I tried to pay", "match": "c8", "category": "paraphrase"},
    {"text": "checkout isn't accepting my payment, why", "match": "c8", "category": "paraphrase"},
    {"text": "payment error at checkout, help", "match": "c8", "category": "paraphrase"},
    {"text": "Why did my payment fail to show up in my transaction history even though it succeeded?", "match": None, "category": "trap"},
    {"text": "Why did my payment method get removed automatically after checkout?", "match": None, "category": "trap"},

    # ---- update_card (c9) ----
    {"text": "How do I update my saved credit card details?", "match": "c9", "category": "exact"},
    {"text": "I need to change the card on file for billing", "match": "c9", "category": "paraphrase"},
    {"text": "how can I edit my payment card information", "match": "c9", "category": "paraphrase"},
    {"text": "update card details please", "match": "c9", "category": "paraphrase"},
    {"text": "How do I update my saved shipping address, not my card?", "match": None, "category": "trap"},
    {"text": "How do I remove my saved credit card entirely, not update it?", "match": None, "category": "trap"},

    # ---- invoice (c10) ----
    {"text": "Where can I download my invoice for last month?", "match": "c10", "category": "exact"},
    {"text": "I need last month's bill as a PDF", "match": "c10", "category": "paraphrase"},
    {"text": "how do I get a copy of my recent invoice", "match": "c10", "category": "paraphrase"},
    {"text": "download invoice please", "match": "c10", "category": "paraphrase"},
    {"text": "Where can I download my invoice template to send to my own clients?", "match": None, "category": "trap"},
    {"text": "Where can I download the app on my new phone, not the invoice?", "match": None, "category": "trap"},

    # ---- unrelated (easy true negatives, spread across topics) ----
    {"text": "What's the weather like today?", "match": None, "category": "unrelated"},
    {"text": "Can you recommend a good sci-fi movie?", "match": None, "category": "unrelated"},
    {"text": "What time zone is your support team in?", "match": None, "category": "unrelated"},
    {"text": "Do you have a mobile app for Android?", "match": None, "category": "unrelated"},
    {"text": "What's your company's privacy policy?", "match": None, "category": "unrelated"},
    {"text": "Can I speak to a human agent instead of a bot?", "match": None, "category": "unrelated"},
    {"text": "What languages does your support team speak?", "match": None, "category": "unrelated"},
    {"text": "Is there a student discount available?", "match": None, "category": "unrelated"},
    {"text": "How do I integrate your API with my own app?", "match": None, "category": "unrelated"},
    {"text": "What are your office hours?", "match": None, "category": "unrelated"},
]


def load_dataset():
    return CANONICAL_QUERIES, TEST_QUERIES


if __name__ == "__main__":
    canon, tests = load_dataset()
    print(f"Canonical (cached) queries: {len(canon)}")
    print(f"Test queries: {len(tests)}")
    from collections import Counter
    print("Category breakdown:", Counter(t["category"] for t in tests))
