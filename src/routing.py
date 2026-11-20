"""Intent -> domain -> support team mapping.

CLINC150 groups its 150 intents into 10 domains. We add an 11th domain,
"appliance", with 6 hand-written intents (Turkish + English).
Each domain is routed to one team of the (fictional) support center.
"""
from __future__ import annotations

from .config import OOS_LABEL

DOMAINS: dict[str, list[str]] = {
    "banking": ["transfer", "transactions", "balance", "freeze_account", "pay_bill",
                "bill_balance", "bill_due", "interest_rate", "routing", "min_payment",
                "order_checks", "pin_change", "report_fraud", "account_blocked",
                "spending_history"],
    "credit_cards": ["credit_score", "report_lost_card", "credit_limit", "rewards_balance",
                     "new_card", "application_status", "card_declined", "international_fees",
                     "apr", "redeem_rewards", "credit_limit_change", "damaged_card",
                     "replacement_card_duration", "improve_credit_score", "expiration_date"],
    "kitchen_dining": ["recipe", "restaurant_reviews", "calories", "nutrition_info",
                       "restaurant_suggestion", "ingredients_list", "ingredient_substitution",
                       "cook_time", "food_last", "meal_suggestion", "restaurant_reservation",
                       "confirm_reservation", "how_busy", "cancel_reservation",
                       "accept_reservations"],
    "home": ["shopping_list", "shopping_list_update", "next_song", "play_music",
             "update_playlist", "todo_list", "todo_list_update", "calendar",
             "calendar_update", "what_song", "order", "order_status", "reminder",
             "reminder_update", "smart_home"],
    "auto_commute": ["traffic", "directions", "gas", "gas_type", "distance",
                     "current_location", "mpg", "oil_change_when", "oil_change_how",
                     "jump_start", "uber", "schedule_maintenance", "last_maintenance",
                     "tire_pressure", "tire_change"],
    "travel": ["book_flight", "book_hotel", "car_rental", "travel_suggestion", "travel_alert",
               "travel_notification", "carry_on", "timezone", "vaccines", "translate",
               "flight_status", "international_visa", "lost_luggage", "plug_type",
               "exchange_rate"],
    "utility": ["time", "alarm", "share_location", "find_phone", "weather", "text",
                "spelling", "make_call", "timer", "date", "calculator",
                "measurement_conversion", "flip_coin", "roll_dice", "definition"],
    "work": ["direct_deposit", "pto_request", "taxes", "payday", "w2", "pto_balance",
             "pto_request_status", "next_holiday", "insurance", "insurance_change",
             "schedule_meeting", "pto_used", "meeting_schedule", "rollover_401k", "income"],
    "small_talk": ["greeting", "goodbye", "tell_joke", "where_are_you_from", "how_old_are_you",
                   "what_is_your_name", "who_made_you", "thank_you", "what_can_i_ask_you",
                   "what_are_your_hobbies", "do_you_have_pets", "are_you_a_bot",
                   "meaning_of_life", "who_do_you_work_for", "fun_fact"],
    "meta": ["change_ai_name", "change_user_name", "cancel", "user_name", "reset_settings",
             "whisper_mode", "repeat", "no", "yes", "maybe", "change_language",
             "change_accent", "change_volume", "change_speed", "sync_device"],
    "appliance": ["appliance_not_draining", "appliance_error_code", "appliance_not_starting",
                  "appliance_leaking", "technician_appointment", "spare_part_order"],
}

TEAMS: dict[str, str] = {
    "appliance": "Appliance Technical Service",
    "banking": "Payments & Billing",
    "credit_cards": "Payments & Billing",
    "home": "Smart Home & Orders",
    "kitchen_dining": "Kitchen & Recipes Assistant",
    "auto_commute": "Mobility & Travel",
    "travel": "Mobility & Travel",
    "utility": "Device & Assistant Settings",
    "meta": "Device & Assistant Settings",
    "work": "HR & Payroll Services",
    "small_talk": "Chatbot (rule-based reply)",
    OOS_LABEL: "Human agent",
}

DOMAIN_OF_INTENT = {i: d for d, intents in DOMAINS.items() for i in intents}
DOMAIN_OF_INTENT[OOS_LABEL] = OOS_LABEL


def domain_of(intent: str) -> str:
    return DOMAIN_OF_INTENT.get(intent, "general")


def route(intent: str) -> tuple[str, str]:
    """Return (domain, team) for an intent; unknown intents go to General Support."""
    d = domain_of(intent)
    return d, TEAMS.get(d, "General Support")
