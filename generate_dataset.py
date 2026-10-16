"""
Generates a synthetic-but-realistic SMS spam/ham dataset (data/spam.csv)
with the standard SMS Spam Collection schema: columns 'label' and 'message'.

NOTE: This script creates a reasonably sized practice dataset so the whole
project works out of the box with no internet access. For better real-world
accuracy, replace data/spam.csv with the actual UCI SMS Spam Collection
dataset (instructions are in README.md).
"""
import csv
import random

random.seed(42)

spam_templates = [
    "WINNER!! You have been selected to receive a £{amt} cash prize! Call {phone} now to claim.",
    "URGENT! Your mobile number has won £{amt} in our weekly draw. Text WIN to {phone} to collect.",
    "Congratulations! You've been selected for a free {item}. Claim now at {url}",
    "FREE entry into our £{amt} weekly competition just text WIN to {phone}",
    "Claim your free {item} today! Limited time offer, click {url} now!!!",
    "You have won a guaranteed £{amt} award. To claim just call {phone} now.",
    "URGENT! We are trying to contact you. Last weekend's draw shows you won £{amt} prize GUARANTEED. Call {phone}",
    "Congrats! 1 year special cinema pass for 2 is yours. Call {phone} now! C Suprman V, Matrix3, StarWars3",
    "SIX chances to win CASH! From 100 to 20,000 pounds. Text CSH11 to {phone}. 18+",
    "This is the 2nd time we have tried to contact you. You have won the £{amt} prize. Call {phone} to claim.",
    "PRIVATE! Your 2024 Account Statement shows you have won a £{amt} reward! Call {phone} to collect.",
    "FreeMsg: Txt CALL to {phone} and claim your reward of 3 hours talk time to use from your phone now!",
    "SMS AUCTION - you have won a {item}! To claim, call {phone} before midnight.",
    "Your credit for £{amt} top-up is ready. Click {url} to activate your bonus now.",
    "Reply now to claim your free {item}. Offer expires today! Visit {url}",
    "Congratulations, your loan of £{amt} has been pre-approved. Apply now at {url}",
    "LAST CHANCE! Claim your free {item} before it expires. Reply YES now.",
    "You have been chosen for a cash reward of £{amt}. Text CLAIM to {phone} immediately.",
    "Hot singles in your area want to chat NOW! Click {url} to view profiles.",
    "Act now! Your account will be suspended unless you verify at {url} within 24 hours.",
    "Limited time: get {item} absolutely free when you sign up at {url} today.",
    "Dear customer, your parcel is on hold. Pay a small fee of £{amt} at {url} to release it.",
    "Your bank account has been locked. Verify your details immediately at {url} to avoid closure.",
    "Cash prize alert! You qualify for £{amt}. Call now {phone}, don't miss out!",
    "XXXMobileMovieClub: To use your credit, click {url} to download content now.",
]

ham_templates = [
    "Hey {name}, are we still meeting for lunch {day}?",
    "Can you pick up some milk on your way home?",
    "I'll be there in about {mins} minutes, traffic is a bit slow.",
    "Happy birthday {name}! Hope you have a wonderful day.",
    "Don't forget we have the team meeting at {time} {day}.",
    "Thanks for helping me move last weekend, really appreciate it.",
    "What time does the movie start {day}?",
    "I finished the report, sending it over to you now.",
    "Can you send me the address for the party on {day}?",
    "Mom says dinner is ready, come down whenever you're free.",
    "Let's catch up {day}, it's been a while.",
    "I'm running a bit late, start without me.",
    "Did you get a chance to look at the homework yet?",
    "The weather looks great for our hike {day}.",
    "I just landed, will call you once I get to the hotel.",
    "Can you remind me to pick up the dry cleaning?",
    "Great job on the presentation today, really well done {name}.",
    "I left my charger at your place, can I grab it {day}?",
    "Let me know if you need anything from the store.",
    "The doctor's appointment got moved to {day}.",
    "Are you free for a call later this evening?",
    "I'm so proud of you {name} for finishing the marathon!",
    "Can we reschedule our meeting to {day} instead?",
    "Just checking in, how are you feeling today?",
    "Thanks for the birthday gift, I absolutely love it.",
    "I'll send you the photos from the trip once I'm home.",
    "Please remember to bring your laptop to class {day}.",
    "The kids had a great time at the park today.",
    "Let's grab coffee sometime this week if you're free.",
    "I think I left my keys in your car, can you check?",
    "Our flight got delayed by {mins} minutes, see you a bit later.",
    "Good morning {name}! Hope you slept well.",
    "Can you help me study for the exam {day}?",
    "I made reservations for {time}, see you at the restaurant.",
    "Just wanted to say thank you for being such a great friend.",
    "The project deadline has been extended to {day}.",
    "I'm heading to the gym {day}, want to join?",
    "Can you water my plants while I'm away next week?",
    "Loved spending time with you today, let's do it again soon.",
    "I need to reschedule our call, something came up at work.",
]

names = ["Sam", "Priya", "Alex", "Maria", "Raj", "Jamie", "Chris", "Nina"]
days = ["tomorrow", "today", "Saturday", "Sunday", "Monday", "next week", "this evening"]
times = ["9am", "noon", "3pm", "6:30pm", "7pm"]
mins_list = ["5", "10", "15", "20", "30"]

amts = ["100", "250", "500", "750", "1000", "2000", "5000", "10,000"]
phones = ["08002986030", "87066", "83600", "08712460324", "09061701461", "80062"]
urls = ["http://claimnow.example.com", "http://freeprize.example.net", "http://win-big.example.org"]
items = ["iPhone", "holiday to Spain", "£100 gift card", "laptop", "smartwatch", "cinema ticket"]

rows = []

def fill(t):
    return t.format(
        amt=random.choice(amts),
        phone=random.choice(phones),
        url=random.choice(urls),
        item=random.choice(items),
    )

# Generate spam messages (with variation)
for _ in range(260):
    t = random.choice(spam_templates)
    rows.append(("spam", fill(t)))

# Generate ham messages (with variation so they aren't all identical strings)
def fill_ham(t):
    return t.format(
        name=random.choice(names),
        day=random.choice(days),
        time=random.choice(times),
        mins=random.choice(mins_list),
    )

for _ in range(340):
    t = random.choice(ham_templates)
    rows.append(("ham", fill_ham(t)))

random.shuffle(rows)

with open("data/spam.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "message"])
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to data/spam.csv")
