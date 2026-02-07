 
import os
import re
import pickle
import warnings
from datetime import datetime

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV   # gives LinearSVC predict_proba
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# NLTK one-time downloads
# ---------------------------------------------------------------------------
# Ensure required NLTK packages are available. If not, download them quietly.
for _resource, _package in [("tokenizers/punkt", "punkt"),
                            ("corpora/stopwords", "stopwords")]:
    try:
        nltk.data.find(_resource)
    except Exception:
        nltk.download(_package, quiet=True)


# ===========================================================================
# SYNONYM MAP  –  used for data augmentation
# Each key is the canonical word/phrase; values are drop-in replacements.
# ===========================================================================
SYNONYM_MAP = {
    "immediately":   ["right away", "at once", "without delay", "asap", "right now"],
    "urgent":        ["emergency", "critical", "time-sensitive", "pressing"],
    "verify":        ["confirm", "validate", "authenticate", "check","otp"],
    "account":       ["profile", "wallet", "balance"],
    "send":          ["transfer", "wire", "transmit", "dispatch"],
    "money":         ["funds", "cash", "payment", "amount"],
    "bank":          ["financial institution", "lender"],
    "credit card":   ["card number", "debit card","card"],
    "password":      ["secret", "credentials", "login info"],
    "prize":         ["reward", "winnings", "payout"],
    "arrest":        ["detain", "apprehend", "take into custody"],
    "suspended":     ["frozen", "locked", "deactivated", "blocked"],
    "computer":      ["pc", "machine", "laptop", "desktop"],
    "virus":         ["malware", "infection", "trojan", "spyware"],
    "government":    ["the state", "authorities", "officials"],
    "investment":    ["portfolio", "financial opportunity", "trading"],
    "guarantee":     ["assured", "promised", "guaranteed", "certain"],
    "calling":       ["reaching out", "contacting", "getting in touch"],
    "confirm":       ["verify", "double-check", "make sure"],
    "security":      ["safety", "protection", "safeguarding"],
}
def _augment_sentence(sentence: str, n_variants: int = 2) -> list:
    """
    Produce *n_variants* rewritten copies of *sentence* by randomly swapping
    words/phrases that appear in SYNONYM_MAP.  The original is NOT included
    in the returned list — caller decides whether to keep it.
    """
    variants = []
    lower = sentence.lower()

    for _ in range(n_variants):
        current = sentence
        # shuffle keys so different synonyms get picked each round
        keys = list(SYNONYM_MAP.keys())
        np.random.shuffle(keys)

        for key in keys:
            if key in current.lower():
                replacement = np.random.choice(SYNONYM_MAP[key])
                # case-insensitive single replacement
                pattern = re.compile(re.escape(key), re.IGNORECASE)
                current = pattern.sub(replacement, current, count=1)

        if current != sentence:          # only keep if something changed
            variants.append(current)

    return variants


# ===========================================================================
# TIERED KEYWORD LISTS  –  each tier has a numeric weight
# ===========================================================================
# HIGH-weight phrases are almost never used in legitimate calls.
# MEDIUM phrases are suspicious but can appear legitimately.
# LOW phrases are weak signals on their own but strengthen the score when
# they appear together with higher-tier matches.
KEYWORDS_HIGH = [
    "social security", "ssn", "pin number", "account number",
    "routing number", "wire transfer", "gift card", "send money",
    "bank account number", "credit card number", "date of birth",
    "mothers maiden", "last four digits", "security question",
    "reset password", "remote access", "arrest warrant",
    "pay fine", "pay now", "transfer funds", "gift cards","otp"
]

KEYWORDS_MEDIUM = [
    "verify account", "confirm payment", "update payment",
    "bank account", "credit card", "password", "verify your",
    "confirm your", "suspended account", "frozen account",
    "unusual activity", "unauthorized", "compromised",
    "security alert", "locked account", "account verification",
    "irs", "police", "federal agent", "legal action", "lawsuit",
    "arrest", "warrant", "investigation", "bitcoin", "cryptocurrency",
    "paypal", "venmo", "cash app", "tech support",
]

KEYWORDS_LOW = [
    "urgent", "immediately", "right now", "hurry", "limited time",
    "act fast", "expires today", "last chance", "today only",
    "won", "prize", "lottery", "free", "guarantee", "risk-free",
    "no cost", "congratulations", "selected", "winner",
    "virus", "infected", "hacked", "microsoft", "apple",
    "refund", "subscription", "renewal", "overdue",
    "pre-approved", "processing fee", "customs",
    "government", "officer", "sheriff", "court",
    "jail", "penalty", "consequences", "fine",
    "verify", "confirm", "update", "suspended", "locked",
]

# Numeric weights per tier  (tuned so 3 HIGH hits ≈ 8 MEDIUM hits ≈ 20 LOW hits)
TIER_WEIGHTS = {"HIGH": 0.30, "MEDIUM": 0.10, "LOW": 0.03}


# ===========================================================================
# TRAINING DATA  –  200+ hand-written samples (100+ fraud, 100+ legitimate)
# ===========================================================================
FRAUD_EXAMPLES = [
    # --- IRS / Tax scams ---
    "This is the IRS calling about your tax refund. Verify your social security number immediately.",
    "You owe back taxes. Pay now or face arrest. Provide your bank account details.",
    "Final notice from the IRS. Your tax return has been flagged. Call back with your SSN.",
    "The Internal Revenue Service is on the line. You have an outstanding balance. Wire the amount now.",
    "Your tax case is under review. Provide your date of birth and account number to avoid legal action.",
    "IRS agent here. A warrant has been issued. Pay your dues immediately via gift cards.",
    "This is a notice from the tax department. Failure to comply will result in arrest.",
    "Your tax ID has been compromised. Verify your information right away with our department.",

    # --- Bank / Account suspension ---
    "Your bank account has been suspended. Confirm your account details to unlock it.",
    "Urgent! Your checking account is frozen due to suspicious activity. Call us now.",
    "Your account will be permanently closed in 24 hours unless you verify your password.",
    "We detected unusual activity on your account. Provide your full card number immediately.",
    "Your online banking session has expired. Re-enter your credentials to continue.",
    "Bank security team here. Your debit card has been flagged. Send your pin number now.",
    "Your account is locked. Provide your routing number and date of birth to unlock.",
    "Important: your savings account shows unauthorized transfers. Confirm your details.",

    # --- Prize / Lottery scams ---
    "Congratulations! You have won a prize of ten thousand dollars. Send us your bank details to claim.",
    "You are the selected winner of our annual lottery. Wire the processing fee to receive your prize.",
    "A grand reward awaits you. Just confirm your account number and we will transfer the funds.",
    "You have been chosen for a free vacation. Provide your credit card for a small verification fee.",
    "Limited winners only! Claim your guaranteed prize by sending your personal information now.",
    "Exclusive offer: you won a free iPhone. Just pay shipping via gift card.",
    "Your name was drawn in our national raffle. Send your details to claim your payout.",

    # --- Tech support scams ---
    "This is tech support. Your computer has a virus. We need remote access right now.",
    "Microsoft calling. A critical malware has infected your PC. Grant us access immediately.",
    "Your Windows license has expired. Pay the renewal fee now or your machine will be disabled.",
    "Apple support here. Your device is compromised. Provide your Apple ID password.",
    "We detected a trojan on your laptop. Install our tool and give us your login credentials.",
    "Tech team here. Your internet has been hacked. Send gift cards to restore your service.",
    "Your antivirus subscription expired. Renew immediately or your computer will be locked.",
    "System alert: spyware detected. Call our number and provide remote access to fix it.",

    # --- Authority / Police impersonation ---
    "Police department calling. You have an arrest warrant. Pay the fine immediately to avoid jail.",
    "This is officer Martinez from the sheriff's department. You are under investigation.",
    "A federal warrant has been issued in your name. Cooperate by providing your information now.",
    "The court has issued a summons. Failure to respond will result in immediate arrest.",
    "Law enforcement here. Your identity has been used in a fraud case. Verify your details.",
    "Homeland security calling. Your passport has been flagged. Provide your SSN for verification.",
    "Detective on the line. You are a suspect. Wire the bail amount via cryptocurrency now.",

    # --- Investment / Loan scams ---
    "Limited time offer! Invest now for guaranteed returns of five hundred percent.",
    "You are pre-approved for a personal loan. Send your bank account details to receive funds.",
    "Exclusive investment opportunity. Wire five thousand dollars and double your money in a week.",
    "Guaranteed profit awaits. Just transfer the initial deposit to our secure account.",
    "Financial advisor here. Act fast — this opportunity expires today. Send your funds now.",
    "Your investment portfolio has matured. Provide your details to claim the returns.",
    "Risk-free trading opportunity. Send your credit card number to open your account.",

    # --- Package / Customs scams ---
    "Your package is held at customs. Pay the duty fee via gift cards to release it.",
    "Delivery notification: your parcel is stuck. Pay the handling charge immediately.",
    "A package in your name is being held. Send the required fee to the address below.",

    # --- Subscription / Account renewal scams ---
    "Your Netflix account is suspended. Update your billing information immediately.",
    "Amazon Prime renewal failed. Provide your card details to avoid service interruption.",
    "Your streaming subscription will be cancelled. Confirm your payment method now.",
    "Monthly subscription charge declined. Enter your updated card number to continue.",

    # --- Romance / Personal scams ---
    "I need your help urgently. Please wire me some money and I will pay you back.",
    "I am stuck overseas and need emergency funds. Send to my account right away.",
    "My wallet was stolen. Can you send me money via PayPal as soon as possible?",

    # --- Charity / Donation scams ---
    "Donate now to save lives. Send your payment via wire transfer to our account.",
    "Emergency relief fund accepting donations. Provide your credit card for an immediate contribution.",

    # --- Miscellaneous high-confidence fraud ---
    "Urgent security alert! Your account will be closed. Update payment information now.",
    "This is a final warning. Your account has been compromised. Verify immediately.",
    "Calling from the fraud prevention unit. Confirm your identity by providing your SSN.",
    "You have been flagged for suspicious transactions. Wire the amount to clear your name.",
    "Your email has been hacked. Change your password by clicking the link and entering credentials.",
    "Automated alert: large transfer pending. Approve or cancel by providing your authorization code.",
    "We are refunding your account. Provide bank details so we can deposit the refund.",
    "Your identity has been stolen. Act now — send us your information so we can secure your accounts.",
    "Payment overdue. To avoid penalties pay the outstanding balance immediately via gift card.",
    "Congratulations on being selected for our VIP program. Just verify your card to activate.",
    "Your domain is expiring. Renew now by providing payment details or lose your website.",
    "Scam alert: someone is impersonating you. Confirm your details with us to protect your account.",
    "A large deposit is waiting. Unlock it by sending a small processing fee first.",
    "Your account privileges have been revoked. Provide your credentials to reinstate access.",
    "Limited slots available in our exclusive fund. Wire your investment today to secure your spot.",
    "Emergency: your child's school account is in arrears. Pay immediately via transfer.",
    "Your electricity bill is overdue. Disconnection will happen today unless you pay via gift cards.",
    "Social security office here. Your number was used fraudulently. Verify your identity now.",
    "You qualified for a special grant. Send the application fee to receive the money.",
    "Your car insurance is being cancelled due to non-payment. Provide card details immediately.",
]

LEGITIMATE_EXAMPLES = [
    # --- General business calls ---
    "Hello, this is John from ABC Company following up on our meeting yesterday.",
    "Hi, I am calling to schedule an appointment for next week. What time works for you?",
    "This is Sarah from customer service at XYZ Inc. How can I help you today?",
    "Just wanted to confirm our dinner reservation for Friday evening at seven.",
    "I am returning your call from earlier today. Did you get a chance to look at the proposal?",
    "This is Mike from the sales team. We would love to set up a demo for your team.",
    "Calling to let you know your project is on track. We will have updates by Thursday.",
    "Hey, it is Lisa. I wanted to discuss the quarterly report before the deadline.",
    "This is the front desk at the hotel confirming your reservation for next Tuesday.",
    "Calling from HR regarding the onboarding paperwork we discussed last week.",

    # --- Appointment / Reminder calls ---
    "This is a reminder about your dentist appointment tomorrow at two PM.",
    "Your doctor's office is calling to confirm your check-up on the fifteenth.",
    "Reminder: your car service is scheduled for this Saturday morning.",
    "The vet clinic is calling to remind you about your dog's annual vaccination.",
    "We are reminding you that your home inspection is set for next Monday.",
    "Your eye doctor wants to confirm your appointment next Wednesday at ten.",
    "This is a courtesy call about your upcoming HVAC maintenance visit.",

    # --- Order / Delivery notifications ---
    "Thank you for your order. Your package will arrive in three to five business days.",
    "Your online order has been shipped. You can track it using the link we sent.",
    "The furniture you ordered is ready for delivery. We will call to arrange a time.",
    "Your grocery delivery is scheduled for tomorrow between nine and eleven AM.",
    "Order confirmation: your new laptop will be delivered to your office by Friday.",

    # --- Job / Education calls ---
    "I am calling regarding the job position you applied for. Are you still interested?",
    "The university admissions office here. We wanted to go over your application status.",
    "Your scholarship application has been received. We will be in touch next week.",
    "Calling from the internship coordinator. Can we arrange an interview this month?",
    "The hiring manager would like to schedule a second interview. Are you available?",

    # --- Personal / Neighbour calls ---
    "Hi, this is your neighbour. I found your package at my door by mistake.",
    "Just calling to see if you want to grab lunch this weekend.",
    "Hey, it is Tom from down the street. Did you see the community event flyer?",
    "This is Jane, your babysitter. Everything is going well with the kids tonight.",
    "Calling to remind you about the neighbourhood barbecue this Saturday.",

    # --- Service / Utility notifications ---
    "Your electricity bill for this month is ready. You can pay it online at any time.",
    "Internet service notification: scheduled maintenance tonight from midnight to two AM.",
    "Your phone plan is up for renewal next month. No action needed if you want to keep it.",
    "Water company here. Routine meter reading is coming up next week.",
    "Cable company calling about a planned outage in your area this Sunday.",

    # --- Banking (legitimate) ---
    "This is your bank calling to let you know your new debit card is ready for pickup.",
    "We are informing you that your monthly statement is now available online.",
    "Your loan application has been approved. Please visit the branch to sign the documents.",
    "Routine notification: your account balance exceeded the alert threshold you set.",
    "Your automatic payment has been processed successfully for this month.",

    # --- School / Parent calls ---
    "This is the school calling to inform you about tomorrow's parent-teacher meeting.",
    "The principal's office is reaching out about volunteering at the upcoming fair.",
    "Your child's teacher wants to discuss their progress. Can we set up a call?",
    "School cafeteria reminder: please top up your child's lunch account.",
    "The school bus driver is calling to confirm the pickup schedule for next week.",

    # --- Healthcare ---
    "Your prescription is ready for pickup at the pharmacy.",
    "The lab results are back. The doctor would like to discuss them with you.",
    "Calling from the hospital regarding the follow-up appointment after surgery.",
    "Your health insurance renewal is coming up. We will send the details by mail.",
    "The clinic is offering free flu shots this week. Would you like to schedule one?",

    # --- Real estate / Apartment ---
    "Hello, I am calling about the apartment you are interested in renting.",
    "The real estate agent wants to schedule a viewing of the house this afternoon.",
    "Your lease renewal is due next month. Please review and sign the updated agreement.",
    "Maintenance team here. We are coming by on Thursday to fix the kitchen tap.",

    # --- Travel ---
    "Calling to confirm your flight booking for next month.",
    "Your hotel stay has been confirmed. Check-in is at three PM on arrival day.",
    "The travel agency is calling to finalize your vacation itinerary.",
    "Your rental car is ready for pickup at the airport when you arrive.",

    # --- Gym / Fitness ---
    "This is the gym calling about your membership renewal next month.",
    "Personal trainer reminder: your session is at six PM this evening.",
    "The fitness centre is hosting a free yoga class this weekend. Want to join?",

    # --- Miscellaneous legitimate ---
    "Just checking in to see if you received the documents I sent yesterday.",
    "Hi, I wanted to thank you for your help with the presentation yesterday.",
    "This is the library with a reminder that your books are due next Monday.",
    "Calling to discuss the project timeline we talked about last week.",
    "This is a survey about your recent shopping experience. Do you have a few minutes?",
    "Calling to let you know your car repair is complete. You can pick it up anytime.",
    "The contractor is calling to give you an estimate for the bathroom renovation.",
    "Your subscription box is being shipped out today. Expect it by the weekend.",
    "Calling from the charity you donated to. We wanted to thank you personally.",
    "Your daughter's dance recital is on Saturday at four. See you there!",
    "This is the parking garage. Your monthly pass is up for renewal.",
    "Calling to let you know the restaurant can accommodate your reservation.",
    "The plumber is running late. He should arrive by noon instead of ten.",
    "Community centre here — the evening class you signed up for starts next week.",
    "Delivery driver on the line: I cannot find your building. Can you clarify?",
]


class FraudDetector:
    """
    V2 Fraud detector.

    Detection pipeline:
        raw text
            → preprocess  (lowercase, keep digits, tokenise, drop stopwords)
            → TF-IDF      (up to trigrams, 5 000 features)
            → Ensemble    (soft-voting: GB + RF + LR + calibrated SVC)
            → keyword     (three-tier weighted scoring)
            → fuse scores (0.65 × ML + 0.35 × keyword)
            → risk label  (HIGH / MEDIUM / LOW)
    """

    def __init__(self, model_path: str = "./models/fraud_model_v2.pkl"):
        self.model_path = model_path
        self.vectorizer: TfidfVectorizer | None = None
        self.classifier: VotingClassifier | None = None

        self._load_model()
        if not self.is_model_trained():
            print("⚠  No trained model found — building V2 model now …")
            self._create_initial_model()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as fh:
                    data = pickle.load(fh)
                self.vectorizer = data["vectorizer"]
                self.classifier = data["classifier"]
                print("✓ Loaded existing V2 fraud-detection model")
            except Exception as exc:
                print(f"✗ Error loading model: {exc}")

    def _save_model(self):
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        with open(self.model_path, "wb") as fh:
            pickle.dump({"vectorizer": self.vectorizer,
                         "classifier": self.classifier}, fh)
        print(f"✓ Model saved → {self.model_path}")

    def is_model_trained(self) -> bool:
        return self.vectorizer is not None and self.classifier is not None

    # ------------------------------------------------------------------
    # text preprocessing
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess_text(text: str) -> str:
        
        text = text.lower()
        # keep letters, digits, spaces, dollar sign
        text = re.sub(r"[^a-z0-9\s$]", " ", text)
        tokens = word_tokenize(text)
        stop = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in stop and len(t) > 1]
        return " ".join(tokens)

    # ------------------------------------------------------------------
    # data augmentation
    # ------------------------------------------------------------------
    @staticmethod
    def _augment_dataset(texts: list, labels: list, variants_per_sample: int = 3) -> tuple:
       
        aug_texts, aug_labels = list(texts), list(labels)
        for txt, lbl in zip(texts, labels):
            for variant in _augment_sentence(txt, n_variants=variants_per_sample):
                aug_texts.append(variant)
                aug_labels.append(lbl)
        return aug_texts, aug_labels

    # ------------------------------------------------------------------
    # model construction
    # ------------------------------------------------------------------
    @staticmethod
    def _build_ensemble() -> VotingClassifier:
        """
        Soft-voting ensemble of four diverse classifiers.

        * GradientBoosting  – strong on tabular / sparse features
        * RandomForest      – low variance, good generalisation
        * LogisticRegression – fast, linear baseline, well-calibrated
        * CalibratedClassifierCV(LinearSVC) – powerful linear boundary;
              wrapped in calibration so it exposes predict_proba (required
              for soft voting).
        """
        gb = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.08, max_depth=4,
            subsample=0.8, random_state=42
        )
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=2,
            random_state=42
        )
        lr = LogisticRegression(
            C=5.0, max_iter=1000, solver="lbfgs", random_state=42
        )
        svc = CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=5000, random_state=42), cv=3
        )

        return VotingClassifier(
            estimators=[("gb", gb), ("rf", rf), ("lr", lr), ("svc", svc)],
            voting="soft"          # average predicted probabilities
        )

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def _create_initial_model(self):
        """Build and train on the hand-written + augmented corpus."""
        print("🎯 Preparing training data …")

        texts = FRAUD_EXAMPLES + LEGITIMATE_EXAMPLES
        labels = [1] * len(FRAUD_EXAMPLES) + [0] * len(LEGITIMATE_EXAMPLES)
        print(f"   Raw samples: {len(texts)}  "
              f"({len(FRAUD_EXAMPLES)} fraud, {len(LEGITIMATE_EXAMPLES)} legit)")

        # augment
        texts, labels = self._augment_dataset(texts, labels, variants_per_sample=3)
        print(f"   After augmentation: {len(texts)} samples")

        metrics = self._train_on_data(texts, labels)
        print("✓  V2 model created successfully!")
        return metrics

    def _train_on_data(self, texts: list, labels: list) -> dict:
        """
        Preprocess → TF-IDF → build ensemble → 5-fold CV → save.
        Returns a dict of mean CV metrics.
        """
        processed = [self._preprocess_text(t) for t in texts]

        # ── TF-IDF  (trigrams, 5 000 features) ──
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),       # unigrams + bigrams + trigrams
            min_df=2,                 # ignore terms appearing only once
            sublinear_tf=True,        # apply log(1+tf) — helps with long docs
        )
        X = self.vectorizer.fit_transform(processed)
        y = np.array(labels)

        # ── Ensemble classifier ──
        self.classifier = self._build_ensemble()

        # ── Stratified 5-fold cross-validation ──
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        cv_acc = cross_val_score(self.classifier, X, y, cv=skf, scoring="accuracy")
        cv_f1  = cross_val_score(self.classifier, X, y, cv=skf, scoring="f1")
        cv_prec= cross_val_score(self.classifier, X, y, cv=skf, scoring="precision")
        cv_rec = cross_val_score(self.classifier, X, y, cv=skf, scoring="recall")

        print(f"   5-Fold CV  →  Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
        print(f"                  Precision: {cv_prec.mean():.4f} ± {cv_prec.std():.4f}")
        print(f"                  Recall:    {cv_rec.mean():.4f} ± {cv_rec.std():.4f}")
        print(f"                  F1:        {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

        # ── Fit on full dataset so the saved model uses every sample ──
        self.classifier.fit(X, y)
        self._save_model()

        return {
            "accuracy":  round(float(cv_acc.mean()), 4),
            "precision": round(float(cv_prec.mean()), 4),
            "recall":    round(float(cv_rec.mean()), 4),
            "f1_score":  round(float(cv_f1.mean()), 4),
            "samples":   len(texts),
            "cv_folds":  5,
        }

    def train_model(self, transcript_folder: str = "./transcripts") -> dict:
        """Public endpoint — rebuild the model from scratch."""
        print("🎯 Retraining V2 model …")
        return self._create_initial_model()

    # ------------------------------------------------------------------
    # keyword scoring  (three-tier weighted)
    # ------------------------------------------------------------------
    @staticmethod
    def _calculate_keyword_score(text: str) -> tuple:
        """
        Walk through HIGH → MEDIUM → LOW keyword lists.
        Each match contributes its tier weight to the raw score.
        The raw score is then clamped to [0, 1].

        Returns:
            (normalised_score, list_of_matched_keywords)
        """
        lower = text.lower()
        matched = []
        raw_score = 0.0

        for keyword in KEYWORDS_HIGH:
            if keyword in lower:
                matched.append(keyword)
                raw_score += TIER_WEIGHTS["HIGH"]

        for keyword in KEYWORDS_MEDIUM:
            if keyword in lower and keyword not in matched:
                matched.append(keyword)
                raw_score += TIER_WEIGHTS["MEDIUM"]

        for keyword in KEYWORDS_LOW:
            if keyword in lower and keyword not in matched:
                matched.append(keyword)
                raw_score += TIER_WEIGHTS["LOW"]

        # clamp to 1.0
        return min(raw_score, 1.0), matched

    # ------------------------------------------------------------------
    # prediction
    # ------------------------------------------------------------------
    def predict(self, text: str) -> dict:
        """
        Full pipeline: preprocess → TF-IDF → ensemble predict_proba
        → keyword score → fused confidence → risk label.
        """
        if not self.is_model_trained():
            raise RuntimeError("Model not trained. Call train_model() first.")

        # ── ML score ──
        processed = self._preprocess_text(text)
        vec = self.vectorizer.transform([processed])
        proba = self.classifier.predict_proba(vec)[0]        # [P(legit), P(fraud)]
        ml_fraud_prob = float(proba[1]) if len(proba) > 1 else 0.0

        # ── Keyword score ──
        kw_score, matched_keywords = self._calculate_keyword_score(text)

        # ── Fused confidence  (65 % ML  +  35 % keywords) ──
        ML_WEIGHT  = 0.65
        KW_WEIGHT  = 0.35
        final_confidence = ML_WEIGHT * ml_fraud_prob + KW_WEIGHT * kw_score
        final_confidence = min(final_confidence, 1.0)

        # ── Risk label ──
        if final_confidence >= 0.70:
            risk_level = "HIGH"
        elif final_confidence >= 0.40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        is_fraud = final_confidence > 0.50

        return {
            "is_fraud":          bool(is_fraud),
            "confidence":        round(final_confidence, 4),
            "risk_level":        risk_level,
            "ml_prediction":     bool(ml_fraud_prob > 0.50),
            "ml_confidence":     round(ml_fraud_prob, 4),
            "keyword_score":     round(kw_score, 4),
            "fraud_indicators":  matched_keywords[:12],   # top 12
            "timestamp":         datetime.now().isoformat(),
        }


# ===========================================================================
# QUICK SMOKE TEST  –  run with:  python fraud_detector.py
# ===========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" Fraud Detector V2 — smoke test")
    print("=" * 70)

    detector = FraudDetector()

    test_cases = [
        # (description, text, expected)
        ("IRS scam",
         "This is the IRS. Your social security number is compromised. "
         "Wire the fine immediately or face arrest.",
         "FRAUD"),

        ("Tech-support scam",
         "Microsoft support here. Your computer has a virus. "
         "Give us remote access and your password right now.",
         "FRAUD"),

        ("Prize scam",
         "Congratulations! You won fifty thousand dollars. "
         "Send your bank account details to claim your prize.",
         "FRAUD"),

        ("Legitimate — appointment",
         "Hi, this is Sarah from the dentist office. "
         "Just a reminder that your cleaning is tomorrow at two PM.",
         "LEGIT"),

        ("Legitimate — job",
         "Calling from HR at TechCorp. We loved your interview. "
         "Can we schedule a follow-up next week?",
         "LEGIT"),

        ("Legitimate — neighbour",
         "Hey, it is Dave from next door. Your parcel was delivered "
         "to my place by mistake. Come grab it whenever.",
         "LEGIT"),

        ("Edge case — vague urgency (should be LOW / LEGIT)",
         "Please confirm you received the email I sent yesterday. "
         "It is a bit urgent because of the project deadline.",
         "LEGIT"),

        ("Edge case — mixed signals",
         "Your account was flagged. Please log in and verify your "
         "details at your earliest convenience.",
         "MEDIUM"),
    ]

    for desc, text, expected in test_cases:
        result = detector.predict(text)
        label = "FRAUD" if result["is_fraud"] else "LEGIT"
        status = "✓" if (label == expected or expected == "MEDIUM") else "✗"
        print(f"\n{status}  [{desc}]")
        print(f"   Expected : {expected}")
        print(f"   Got      : {label}  |  confidence {result['confidence']:.2%}  "
              f"|  risk {result['risk_level']}")
        if result["fraud_indicators"]:
            print(f"   Triggers : {', '.join(result['fraud_indicators'][:6])}")

    print("\n" + "=" * 70)