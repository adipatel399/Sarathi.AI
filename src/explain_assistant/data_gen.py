"""Synthetic training data for Sarathi.AI.

Each case has several message phrasings; one alternate English phrasing per case is reserved for
the test split, so the eval measures generalisation to unseen wording, not memorisation.
"""

import argparse
import json
import random
from pathlib import Path

from explain_assistant.prompt import build_messages

BANKS = [
    ("SBI", "SBIINB", "1800 1234"),
    ("HDFC Bank", "HDFCBK", "1800 202 6161"),
    ("ICICI Bank", "ICICIB", "1800 1080"),
    ("Punjab National Bank", "PNBSMS", "1800 180 2222"),
    ("Bank of Baroda", "BOBTXN", "1800 5700"),
    ("Canara Bank", "CANBNK", "1800 1030"),
    ("Axis Bank", "AXISBK", "1860 419 5555"),
]
UTILITIES = [
    ("BESCOM", "BESCOM"), ("Tata Power-DDL", "TPDDLS"), ("MSEDCL", "MSEDCL"),
    ("UPPCL", "UPPCLS"), ("Adani Electricity", "ADANIE"), ("TNPDCL", "TNPDCL"),
]
SHOPS = [("Amazon", "AMAZON"), ("Flipkart", "FLPKRT"), ("India Post", "INDPST"), ("Meesho", "MEESHO")]
HOSPITALS = [("Apollo Hospitals", "APOLLO"), ("Max Healthcare", "MAXHLT"), ("Fortis", "FORTIS"), ("Manipal Hospitals", "MANIPL")]
PREFIXES = ["VM", "VK", "AD", "JD", "BP", "AX", "JM"]
MONTHS = [("January", "जनवरी"), ("February", "फ़रवरी"), ("March", "मार्च"), ("April", "अप्रैल"), ("May", "मई"),
          ("June", "जून"), ("July", "जुलाई"), ("August", "अगस्त"), ("September", "सितंबर"),
          ("October", "अक्टूबर"), ("November", "नवंबर"), ("December", "दिसंबर")]
AGENCIES = ["CBI", "Mumbai Police Cyber Cell", "Customs Department", "Narcotics Control Bureau", "TRAI", "Delhi Crime Branch"]
ITEMS = [("mobile cover", "मोबाइल कवर"), ("pressure cooker", "प्रेशर कुकर"), ("reading glasses", "चश्मा"),
         ("BP monitor", "बीपी मशीन"), ("saree", "साड़ी"), ("walking stick", "छड़ी")]
DEPTS = [("Cardiology", "दिल के डॉक्टर"), ("Orthopaedics", "हड्डी के डॉक्टर"), ("Diabetology", "शुगर के डॉक्टर"),
         ("Ophthalmology", "आँख के डॉक्टर"), ("General Medicine", "जनरल फ़िज़िशियन")]
NAMES = ["Sharma", "Verma", "Gupta", "Iyer", "Reddy", "Patel", "Singh", "Mehta", "Nair", "Joshi"]
DOCTORS = ["R. K. Mehta", "Anita Rao", "S. Banerjee", "Farhan Qureshi", "P. Krishnan", "Neha Kapoor"]
SCAMMERS = ["Rahul Sharma", "Vikram Singh", "Rana Pratap", "Amit Kumar", "Priya Verma"]
LINK_WORDS = ["kyc-update", "netbank-verify", "secure-login", "redelivery", "pension-update", "refund-claim"]
LINK_TLDS = [".in", ".com", ".top", ".xyz", ".online", ".info"]
PLACES = ["Sector 21", "Gandhi Nagar", "MG Road", "Civil Lines", "Anna Nagar"]

DRUGS = [
    ("Metformin", "500mg", "शुगर (डायबिटीज़)", "blood sugar (diabetes)", "after food"),
    ("Amlodipine", "5mg", "ब्लड प्रेशर", "blood pressure", "after food"),
    ("Telmisartan", "40mg", "ब्लड प्रेशर", "blood pressure", "after food"),
    ("Atorvastatin", "10mg", "कोलेस्ट्रॉल", "cholesterol", "after food"),
    ("Pantoprazole", "40mg", "एसिडिटी और गैस", "acidity and gas", "before food"),
    ("Paracetamol", "650mg", "बुखार और दर्द", "fever and pain", "after food"),
    ("Thyroxine", "50mcg", "थायरॉइड", "thyroid", "empty stomach"),
    ("Clopidogrel", "75mg", "खून पतला रखने (दिल)", "keeping the blood thin (heart)", "after food"),
    ("Calcium + Vitamin D3", "500mg", "हड्डियों की मज़बूती", "bone strength", "after food"),
]
SCHEDULES = {
    "1-0-1": ("सुबह और रात", "morning and night"),
    "1-0-0": ("सिर्फ़ सुबह", "only in the morning"),
    "0-0-1": ("सिर्फ़ रात को", "only at night"),
    "1-1-1": ("दिन में तीन बार", "three times a day"),
    "0-1-0": ("दोपहर में", "in the afternoon"),
}
FOOD = {
    "after food": ("खाने के बाद", "after food"),
    "before food": ("खाने से पहले", "before food"),
    "empty stomach": ("खाली पेट", "on an empty stomach"),
}


def _money(rng, lo, hi):
    return f"{rng.randint(lo, hi):,}"


def _mobile(rng):
    return f"+91 {rng.choice('9876')}{rng.randint(100000000, 999999999)}"


def _slots(rng):
    bank, bank_hdr, helpline = rng.choice(BANKS)
    utility, util_hdr = rng.choice(UTILITIES)
    shop, shop_hdr = rng.choice(SHOPS)
    hospital, hosp_hdr = rng.choice(HOSPITALS)
    month_en, month_hi = rng.choice(MONTHS)
    item_en, item_hi = rng.choice(ITEMS)
    dept_en, dept_hi = rng.choice(DEPTS)
    day = rng.randint(1, 28)
    return {
        "bank": bank, "bank_hdr": bank_hdr, "helpline": helpline,
        "utility": utility, "util_hdr": util_hdr,
        "shop": shop, "shop_hdr": shop_hdr,
        "hospital": hospital, "hosp_hdr": hosp_hdr,
        "month": month_en, "month_hi": month_hi,
        "date": f"{day:02d}-{MONTHS.index((month_en, month_hi)) + 1:02d}-2026",
        "time": rng.choice(["10:30 AM", "11:00 AM", "4:15 PM", "9:00 AM", "5:30 PM"]),
        "item": item_en, "item_hi": item_hi,
        "dept": dept_en, "dept_hi": dept_hi,
        "l4": f"{rng.randint(0, 9999):04d}",
        "otp": f"{rng.randint(100000, 999999)}",
        "amt": _money(rng, 150, 48000),
        "bal": _money(rng, 2000, 350000),
        "fee": rng.choice(["25", "49", "99", "199", "499", "1,500", "2,999"]),
        "lakh": rng.choice(["5", "10", "25", "35", "50"]),
        "acct": f"{rng.randint(10**9, 10**10 - 1)}",
        "policy": f"{rng.randint(10**8, 10**9 - 1)}",
        "ref": f"{rng.randint(10**11, 10**12 - 1)}",
        "merchant": rng.choice(["Reliance Fresh", "Apollo Pharmacy", "BigBasket", "Paytm", "IRCTC", "Jio Recharge"]),
        "phone": _mobile(rng),
        "upi": f"{rng.choice(['help', 'rahul', 'fast', 'pay', 'money'])}{rng.randint(10, 9999)}@{rng.choice(['ybl', 'okaxis', 'paytm', 'ibl'])}",
        "link": f"http://{rng.choice(['sbi', 'hdfc', 'india-post', 'gov', 'bank', 'pm'])}-{rng.choice(LINK_WORDS)}{rng.choice(LINK_TLDS)}/{rng.randint(1000, 99999)}",
        "agency": rng.choice(AGENCIES),
        "scammer": rng.choice(SCAMMERS),
        "name": rng.choice(NAMES),
        "doctor": rng.choice(DOCTORS),
        "place": rng.choice(PLACES),
    }


def _sender(kind, s, rng):
    prefix = rng.choice(PREFIXES)
    return {
        "bank": f"{prefix}-{s['bank_hdr']}",
        "utility": f"{prefix}-{s['util_hdr']}",
        "shop": f"{prefix}-{s['shop_hdr']}",
        "hospital": f"{prefix}-{s['hosp_hdr']}",
        "lic": f"{prefix}-LICIND",
        "pmkisan": f"{prefix}-PMKISN",
        "itd": f"{prefix}-ITDEPT",
        "mobile": s["phone"] if rng.random() < 0.5 else _mobile(rng),
        "photo": "photo of a document",
    }[kind]


SAFE = dict(verdict="safe", is_dangerous=False, red_flags=[])


def _scam(flags, urgency):
    return dict(doc_type="scam", verdict="scam", is_dangerous=True, urgency=urgency, red_flags=flags)


CASES = [
    dict(
        name="otp_genuine", sender="bank", label=dict(doc_type="otp", urgency="low", **SAFE),
        variants=[
            ("en", "{otp} is your OTP for a transaction of Rs {amt} at {merchant} on your {bank} card XX{l4}. Valid for 10 mins. Do not share it with anyone. -{bank}"),
            ("en", "Dear Customer, OTP for login to {bank} NetBanking is {otp}. Never share your OTP with anyone. Bank never asks for it."),
            ("hi", "{bank}: आपके खाते XX{l4} से Rs {amt} के लेन-देन के लिए OTP {otp} है। यह 10 मिनट तक मान्य है। इसे किसी को न बताएं।"),
            ("hinglish", "{bank} alert: aapka OTP {otp} hai, account XX{l4} ki payment ke liye. Ye OTP kisi ko mat batayein."),
        ],
        exp=("यह {bank} की तरफ़ से आया असली OTP (एक बार का पासवर्ड) संदेश है। यह पासवर्ड सिर्फ़ आपके लिए है।",
             "This is a genuine OTP (one-time password) message from {bank}. The code is only for you."),
        todo=("अगर यह लेन-देन आपने ख़ुद शुरू नहीं किया है तो OTP किसी को न बताएं और बैंक की हेल्पलाइन {helpline} पर फोन करें।",
              "Do not share this OTP with anyone. If you did not start this transaction, call the bank helpline {helpline}."),
    ),
    dict(
        name="debit_alert", sender="bank", label=dict(doc_type="bank_alert", urgency="low", **SAFE),
        variants=[
            ("en", "Rs {amt} debited from A/c XX{l4} on {date} towards {merchant}. Avl Bal: Rs {bal}. If not done by you, call {helpline} -{bank}"),
            ("en", "Your {bank} a/c XX{l4} is debited for INR {amt} on {date} by UPI ref {ref}. Not you? Report at {helpline}."),
            ("hi", "{bank}: आपके खाते XX{l4} से {date} को Rs {amt} कटे हैं। शेष राशि Rs {bal}। अगर आपने नहीं किया तो {helpline} पर कॉल करें।"),
            ("hinglish", "{bank}: A/c XX{l4} se Rs {amt} debit hue {date} ko. Balance Rs {bal}. Aapne nahi kiya to {helpline} par call karein."),
        ],
        exp=("आपके {bank} खाते (आख़िरी अंक {l4}) से {date} को Rs {amt} निकले हैं। यह बैंक का असली सूचना संदेश है।",
             "Rs {amt} was taken out of your {bank} account ending {l4} on {date}. This is a genuine alert from your bank."),
        todo=("अगर यह पैसा आपने ख़ुद नहीं भेजा, तो तुरंत बैंक की हेल्पलाइन {helpline} पर फोन करें।",
              "If you did not make this payment, call the bank helpline {helpline} right away."),
    ),
    dict(
        name="pension_credit", sender="bank", label=dict(doc_type="bank_alert", urgency="low", **SAFE),
        variants=[
            ("en", "Your A/c XX{l4} is credited with Rs {amt} on {date} - PENSION FOR {month}. Avl Bal Rs {bal} -{bank}"),
            ("hi", "{bank}: {month_hi} महीने की पेंशन Rs {amt} आपके खाते XX{l4} में {date} को जमा हो गई है।"),
            ("en", "Dear Customer, INR {amt} credited to your {bank} account XX{l4} on {date} (Pension {month})."),
            ("hinglish", "{bank}: Pension {month} Rs {amt} aapke khata XX{l4} mein {date} ko jama ho gayi hai."),
        ],
        exp=("आपकी {month_hi} महीने की पेंशन Rs {amt} आपके {bank} खाते में आ गई है।",
             "Your pension for {month}, Rs {amt}, has been credited to your {bank} account."),
        todo=("कुछ करने की ज़रूरत नहीं है। पैसे निकालते समय किसी को भी PIN या OTP न बताएं।",
              "Nothing to do. Never tell anyone your PIN or OTP when withdrawing money."),
    ),
    dict(
        name="electricity_bill_genuine", sender="utility", label=dict(doc_type="utility_bill", urgency="medium", **SAFE),
        variants=[
            ("en", "Dear Consumer, your electricity bill for account {acct} is Rs {amt}. Due date: {date}. Pay on the official {utility} website or app. Ignore if already paid."),
            ("en", "{utility}: Last reminder! Bill of Rs {amt} for CA No {acct} is due on {date}. Pay by due date to avoid late payment surcharge."),
            ("hi", "{utility}: उपभोक्ता संख्या {acct} का बिजली बिल Rs {amt} है। भुगतान की अंतिम तिथि {date} है।"),
            ("hinglish", "{utility}: CA {acct} ka bijli bill Rs {amt} hai, {date} tak bharein warna late fee lagegi. Official app se pay karein."),
        ],
        exp=("यह {utility} की तरफ़ से बिजली बिल की असली सूचना है। बिल Rs {amt} का है और {date} तक भरना है।",
             "This is a genuine electricity bill notice from {utility}. The bill is Rs {amt}, due by {date}."),
        todo=("{date} से पहले बिल {utility} की ऑफिशियल ऐप, वेबसाइट या बिजली दफ़्तर में भरें। किसी अनजान नंबर या लिंक पर पैसे न भेजें।",
              "Pay before {date} through the official {utility} app, website or office. Never pay through unknown numbers or links."),
    ),
    dict(
        name="insurance_premium", sender="lic", label=dict(doc_type="insurance_emi", urgency="medium", **SAFE),
        variants=[
            ("en", "Dear Policyholder, premium of Rs {amt} for policy no {policy} is due on {date}. Pay via LIC website, LIC app or nearest branch."),
            ("hi", "LIC: पॉलिसी नंबर {policy} का प्रीमियम Rs {amt} {date} को देय है। LIC ऐप, वेबसाइट या शाखा से भुगतान करें।"),
            ("en", "LIC reminder: Policy {policy} premium Rs {amt} due {date}. Grace period applies. Pay only through official LIC channels."),
            ("hinglish", "LIC: aapki policy {policy} ka premium Rs {amt} {date} tak bharna hai. LIC app ya branch se hi bharein."),
        ],
        exp=("यह LIC की असली याद दिलाने वाली सूचना है। पॉलिसी {policy} का प्रीमियम Rs {amt} {date} तक भरना है।",
             "This is a genuine LIC reminder. The premium of Rs {amt} for policy {policy} is due on {date}."),
        todo=("{date} से पहले LIC ऐप, वेबसाइट या नज़दीकी LIC शाखा से प्रीमियम भरें ताकि पॉलिसी बंद न हो।",
              "Pay the premium before {date} using the LIC app, website or branch so the policy stays active."),
    ),
    dict(
        name="prescription", sender="photo", label=dict(doc_type="prescription", urgency="low", **SAFE),
        variants=["numbered", "dash", "compact", "table"],
    ),
    dict(
        name="life_certificate", sender="bank", label=dict(doc_type="govt_notice", urgency="medium", **SAFE),
        variants=[
            ("en", "Dear Pensioner, please submit your Digital Life Certificate (Jeevan Pramaan) before {date} at your bank branch, post office or the Jeevan Pramaan app to continue receiving pension. -{bank}"),
            ("hi", "{bank}: प्रिय पेंशनभोगी, पेंशन जारी रखने के लिए {date} से पहले अपना जीवन प्रमाण पत्र बैंक शाखा, डाकघर या जीवन प्रमाण ऐप से जमा करें।"),
            ("en", "Reminder from {bank}: Pensioners must submit life certificate by {date}. Visit branch with Aadhaar and PPO number. Bank will never ask OTP on call."),
            ("hinglish", "{bank}: Pensioners dhyan dein, {date} se pehle jeevan pramaan patra bank branch ya post office mein jama karein taaki pension chalu rahe."),
        ],
        exp=("यह पेंशन जारी रखने के लिए जीवन प्रमाण पत्र जमा करने की असली सूचना है। आख़िरी तारीख {date} है।",
             "This is a genuine reminder to submit your life certificate so your pension continues. The last date is {date}."),
        todo=("{date} से पहले आधार और PPO नंबर लेकर बैंक शाखा या डाकघर जाएँ, या परिवार की मदद से जीवन प्रमाण ऐप इस्तेमाल करें।",
              "Before {date}, visit your bank branch or post office with Aadhaar and PPO number, or use the Jeevan Pramaan app with family help."),
    ),
    dict(
        name="pmkisan_credit", sender="pmkisan", label=dict(doc_type="govt_notice", urgency="low", **SAFE),
        variants=[
            ("en", "PM-KISAN: Rs 2000 installment has been credited to your bank account XX{l4} on {date}."),
            ("hi", "पीएम-किसान: Rs 2000 की किस्त {date} को आपके बैंक खाते XX{l4} में जमा कर दी गई है।"),
            ("en", "Dear Farmer, the PM-KISAN installment of Rs 2000 is transferred to your {bank} a/c XX{l4}. Check status on pmkisan.gov.in."),
            ("hinglish", "PM-KISAN: Rs 2000 ki kist aapke bank khate XX{l4} mein {date} ko bhej di gayi hai."),
        ],
        exp=("यह पीएम-किसान योजना की असली सूचना है। Rs 2000 की किस्त आपके बैंक खाते में आ गई है।",
             "This is a genuine PM-KISAN message. The Rs 2000 installment has reached your bank account."),
        todo=("कुछ करने की ज़रूरत नहीं है। किस्त के नाम पर कोई फोन करके OTP या फीस मांगे तो न दें।",
              "Nothing to do. If anyone calls asking for OTP or a fee for this installment, do not give it."),
    ),
    dict(
        name="delivery_update", sender="shop", label=dict(doc_type="delivery_update", urgency="low", **SAFE),
        variants=[
            ("en", "{shop}: Your order of {item} has been shipped and will be delivered by {date}. Track it in the app."),
            ("hi", "{shop}: आपका ऑर्डर ({item_hi}) {date} तक आपके घर पहुँच जाएगा।"),
            ("en", "Out for delivery: your {item} will arrive today. Delivery OTP {otp} - share it only with the delivery agent at your door. -{shop}"),
            ("hinglish", "{shop}: aapka {item} order aaj deliver hoga. Delivery ke waqt hi OTP {otp} batayein."),
        ],
        exp=("यह आपके ऑनलाइन ऑर्डर ({item_hi}) की डिलीवरी की असली सूचना है।",
             "This is a genuine delivery update for your online order ({item})."),
        todo=("सामान घर आने पर ही डिलीवरी वाले को OTP बताएं। कोई फोन पर OTP या पैसे मांगे तो न दें।",
              "Share the delivery OTP only when the parcel is at your door. Never give it over the phone."),
    ),
    dict(
        name="appointment", sender="hospital", label=dict(doc_type="appointment", urgency="low", **SAFE),
        variants=[
            ("en", "Dear {name} ji, your appointment with Dr. {doctor} ({dept}) at {hospital} is confirmed for {date} at {time}. Please arrive 15 minutes early with previous reports."),
            ("hi", "{hospital}: {name} जी, डॉ. {doctor} ({dept}) के साथ आपका अपॉइंटमेंट {date} को {time} बजे पक्का है। पुरानी रिपोर्ट साथ लाएँ।"),
            ("en", "{hospital}: Appointment confirmed. Doctor: Dr. {doctor}, {dept}. Date: {date}, Time: {time}. Reply C to cancel."),
            ("hinglish", "{hospital}: {name} ji, Dr. {doctor} ({dept}) se aapka appointment {date} ko {time} baje confirm hai. Purani reports saath layein."),
        ],
        exp=("{date} को {time} बजे {hospital} में डॉ. {doctor} ({dept_hi}) के साथ आपका अपॉइंटमेंट पक्का हो गया है।",
             "Your appointment with Dr. {doctor} ({dept}) at {hospital} is confirmed for {date} at {time}."),
        todo=("{date} को समय से 15 मिनट पहले पुरानी रिपोर्ट लेकर अस्पताल पहुँचें।",
              "On {date}, reach the hospital 15 minutes early and carry your old reports."),
    ),
    dict(
        name="kyc_scam", sender="mobile", label=_scam(["unknown_sender", "account_block_threat", "suspicious_link"], "high"),
        variants=[
            ("en", "Dear {bank} customer, your KYC has expired and your account will be BLOCKED today. Update KYC immediately: {link}"),
            ("en", "{bank} ALERT: Your NetBanking will be suspended within 24 hrs. Complete PAN update here {link} to avoid blocking."),
            ("hi", "प्रिय ग्राहक, आपका {bank} खाता आज बंद हो जाएगा। तुरंत KYC अपडेट करें: {link}"),
            ("hinglish", "{bank} customer dhyan dein: aapka KYC pending hai, aaj raat account block ho jayega. Turant update karein {link}"),
        ],
        exp=("यह धोखाधड़ी (स्कैम) वाला संदेश है। {bank} कभी SMS लिंक से KYC अपडेट करने या खाता बंद करने की धमकी नहीं देता, और यह एक आम मोबाइल नंबर से आया है।",
             "This is a scam. {bank} never threatens to block your account or asks for KYC updates through an SMS link, and this came from a personal mobile number."),
        todo=("लिंक पर बिल्कुल क्लिक न करें और कोई जानकारी या OTP न दें। शक हो तो ख़ुद बैंक शाखा जाएँ।",
              "Do not click the link or share any details or OTP. If worried, visit your bank branch yourself."),
    ),
    dict(
        name="electricity_cut_scam", sender="mobile", label=_scam(["unknown_sender", "disconnection_threat", "call_this_number"], "high"),
        variants=[
            ("en", "Dear Consumer, your electricity power will be disconnected tonight at 9.30 pm from electricity office because your previous month bill was not updated. Please immediately contact our electricity officer {phone}. Thank you"),
            ("en", "{utility} NOTICE: Your connection will be cut today due to pending bill verification. Call officer {phone} now to avoid disconnection."),
            ("hi", "प्रिय उपभोक्ता, आपका पिछले महीने का बिल अपडेट नहीं हुआ है, आज रात 9:30 बजे आपकी बिजली काट दी जाएगी। तुरंत बिजली अधिकारी से {phone} पर संपर्क करें।"),
            ("hinglish", "Priya upbhokta, aaj raat 9:30 baje aapki bijli kaat di jayegi kyunki pichhla bill update nahi hua. Turant {phone} par sampark karein."),
        ],
        exp=("यह बिजली कटने का झूठा डर दिखाने वाला स्कैम है। बिजली विभाग निजी मोबाइल नंबर से ऐसे संदेश नहीं भेजता। फोन करने पर ये लोग ऐप डलवाकर या OTP लेकर खाते से पैसे निकाल लेते हैं।",
             "This is a fake electricity disconnection scam. Power companies do not send such messages from personal numbers. If you call, they make you install an app or share an OTP and empty your account."),
        todo=("दिए गए नंबर पर फोन न करें। बिल की स्थिति सिर्फ़ बिजली विभाग की ऑफिशियल ऐप या दफ़्तर से जाँचें।",
              "Do not call the number. Check your bill only on the official electricity app or at the office."),
    ),
    dict(
        name="digital_arrest", sender="mobile", label=_scam(["digital_arrest", "impersonates_police", "secrecy_demand", "money_transfer_request"], "high"),
        variants=[
            ("en", "This is {agency}. A parcel in your name containing illegal drugs has been seized. You are under digital arrest. Do not disconnect or tell your family. Call {phone} immediately or a warrant will be issued."),
            ("en", "NOTICE from {agency}: Your Aadhaar is linked to a money laundering case. Join video call on WhatsApp {phone} for verification and transfer your savings to the RBI safe account for checking."),
            ("hi", "यह {agency} से है। आपके आधार से जुड़ा मनी लॉन्ड्रिंग केस दर्ज हुआ है। आप डिजिटल अरेस्ट में हैं, किसी को न बताएं। तुरंत {phone} पर कॉल करें।"),
            ("hinglish", "{agency} se bol rahe hain, aapke naam ka parcel pakda gaya hai. Aap digital arrest mein hain, family ko mat batana, {phone} par video call join karein."),
        ],
        exp=("यह 'डिजिटल अरेस्ट' स्कैम है। पुलिस, CBI या कोई सरकारी विभाग फोन या वीडियो कॉल पर गिरफ़्तारी नहीं करता और न ही पैसे ट्रांसफर करवाता है। 'परिवार को मत बताना' कहना ठगों की पहचान है।",
             "This is a 'digital arrest' scam. Police, CBI or any government agency never arrest anyone over a phone or video call or ask you to transfer money. Telling you to keep it secret from family is a clear sign of fraud."),
        todo=("फोन तुरंत काट दें, कोई पैसा न भेजें और अभी अपने परिवार को बताएं। 1930 साइबर हेल्पलाइन पर शिकायत करें।",
              "Hang up, send no money and tell your family right now. Report it on the 1930 cyber helpline."),
    ),
    dict(
        name="lottery_scam", sender="mobile", label=_scam(["lottery_prize", "advance_fee", "unknown_sender"], "medium"),
        variants=[
            ("en", "Congratulations! Your mobile number has won Rs {lakh} lakh in KBC Lucky Draw. To claim, contact manager {scammer} on WhatsApp {phone} and pay the processing fee."),
            ("hi", "बधाई हो! आपका नंबर KBC लॉटरी में {lakh} लाख रुपये जीता है। इनाम पाने के लिए {phone} पर व्हाट्सऐप करें और प्रोसेसिंग फीस जमा करें।"),
            ("en", "Dear winner, you are selected for a Rs {lakh} lakh cash prize in the Jio lucky draw. Send Rs {fee} registration fee to claim your prize. Contact {phone}."),
            ("hinglish", "Mubarak ho! Aapne {lakh} lakh ki lottery jeeti hai. Inaam lene ke liye Rs {fee} fees bhejein aur {phone} par WhatsApp karein."),
        ],
        exp=("यह लॉटरी का झूठा स्कैम है। आपने कोई लॉटरी नहीं खेली, और असली इनाम के लिए कभी पहले फीस नहीं मांगी जाती।",
             "This is a fake lottery scam. You never entered any lottery, and a real prize never asks you to pay a fee first."),
        todo=("कोई फीस न भेजें और उस नंबर पर संपर्क न करें। संदेश डिलीट करके नंबर ब्लॉक कर दें।",
              "Do not pay any fee or contact that number. Delete the message and block the number."),
    ),
    dict(
        name="upi_pin_scam", sender="mobile", label=_scam(["upi_pin_request", "fake_refund"], "high"),
        variants=[
            ("en", "You have received Rs {amt} PhonePe cashback. Enter your UPI PIN to receive the money in your account."),
            ("en", "Refund of Rs {amt} for your electricity overpayment is pending. Accept the collect request and enter UPI PIN to get the refund."),
            ("hi", "आपको Rs {amt} का कैशबैक मिला है। पैसे अपने खाते में लेने के लिए अपना UPI PIN डालें।"),
            ("hinglish", "Aapke account mein Rs {amt} refund aaya hai, paise lene ke liye request accept karke UPI PIN daalein."),
        ],
        exp=("यह UPI स्कैम है। पैसे पाने के लिए कभी UPI PIN नहीं डालना पड़ता — PIN डालते ही पैसे आपके खाते से कट जाते हैं।",
             "This is a UPI scam. You never need your UPI PIN to receive money — entering the PIN sends money out of your account."),
        todo=("कोई UPI PIN न डालें और आई हुई 'Collect request' को Decline कर दें।",
              "Do not enter your UPI PIN. Decline the collect request."),
    ),
    dict(
        name="family_impersonation", sender="mobile", label=_scam(["impersonates_family", "new_number", "urgent_money_request", "discourages_calling"], "high"),
        variants=[
            ("en", "Hi Papa, this is my new number, my phone fell. Urgent, I need Rs {amt} for hospital, please send to this UPI {upi}. Will return tonight. Don't call, I'm in a meeting."),
            ("hi", "मम्मी, यह मेरा नया नंबर है, पुराना फोन ख़राब हो गया। बहुत ज़रूरी है, Rs {amt} इस UPI {upi} पर अभी भेज दो। अभी फोन मत करना।"),
            ("en", "Uncle namaste, I am your son's friend. He had an accident, send Rs {amt} immediately to {upi} for hospital deposit. Please don't call him now."),
            ("hinglish", "Papa mera phone kharab ho gaya, ye naya number hai. Urgent Rs {amt} bhej do is UPI par {upi}, call mat karna abhi busy hoon."),
        ],
        exp=("यह परिवार वाला बनकर पैसे माँगने का स्कैम है। 'नया नंबर', 'बहुत ज़रूरी' और 'फोन मत करना' — ये ठगी के आम संकेत हैं।",
             "This is a scam where someone pretends to be family. 'New number', 'urgent' and 'don't call' are classic fraud signs."),
        todo=("पैसे न भेजें। पहले अपने बच्चे या रिश्तेदार को उनके पुराने नंबर पर ख़ुद फोन करके पक्का करें।",
              "Do not send money. First call your child or relative on their old number yourself to check."),
    ),
    dict(
        name="parcel_fee_scam", sender="mobile", label=_scam(["suspicious_link", "advance_fee", "urgency_threat"], "medium"),
        variants=[
            ("en", "India Post: Your parcel is held at our warehouse due to incomplete address. Update address and pay Rs {fee} redelivery fee within 24 hours: {link}"),
            ("en", "FedEx: A courier in your name is held by customs. Pay customs duty Rs {amt} to release it: {link}"),
            ("hi", "इंडिया पोस्ट: अधूरे पते की वजह से आपका पार्सल रोका गया है। 24 घंटे में {link} पर पता अपडेट करें और Rs {fee} फीस भरें।"),
            ("hinglish", "Aapka parcel address galat hone ki wajah se ruka hua hai. {link} par Rs {fee} bhar kar address update karein."),
        ],
        exp=("यह पार्सल के नाम पर फीस माँगने वाला स्कैम है। इंडिया पोस्ट या कूरियर कंपनी ऐसे अनजान लिंक पर पैसे नहीं मांगती। लिंक खोलने पर आपके कार्ड या बैंक की जानकारी चुराई जाती है।",
             "This is a parcel fee scam. India Post and courier companies do not collect fees through unknown links. The link steals your card or bank details."),
        todo=("लिंक न खोलें और कोई भुगतान न करें। पार्सल की जानकारी सिर्फ़ ऑफिशियल वेबसाइट या डाकघर से लें।",
              "Do not open the link or pay anything. Check parcel status only on the official website or at the post office."),
    ),
    dict(
        name="pension_scam", sender="mobile", label=_scam(["pension_threat", "otp_request", "unknown_sender"], "high"),
        variants=[
            ("en", "Dear Pensioner, your pension will be stopped from next month as your life certificate is not updated. Update now at {link} and share the OTP with our officer."),
            ("hi", "प्रिय पेंशनभोगी, जीवन प्रमाण पत्र अपडेट न होने से अगले महीने से आपकी पेंशन बंद हो जाएगी। अभी {link} पर अपडेट करें और आया हुआ OTP हमारे अधिकारी को बताएं।"),
            ("en", "Pension Department: Your DA arrears of Rs {amt} are pending. Call {phone} and verify your bank details and OTP to receive the arrears."),
            ("hinglish", "Pensioner ji, aapki pension band hone wali hai. {link} par jeevan pramaan update karein aur OTP officer ko batayein."),
        ],
        exp=("यह पेंशन के नाम पर ठगी का संदेश है। पेंशन विभाग या बैंक कभी OTP या बैंक डिटेल फोन या लिंक पर नहीं मांगता, और यह अनजान नंबर से आया है।",
             "This is a pension scam. The pension office and banks never ask for OTP or bank details over the phone or a link, and this came from an unknown number."),
        todo=("कोई OTP या जानकारी न दें। जीवन प्रमाण पत्र सिर्फ़ बैंक शाखा, डाकघर या ऑफिशियल जीवन प्रमाण ऐप से जमा करें।",
              "Do not share any OTP or details. Submit your life certificate only at the bank branch, post office or the official Jeevan Pramaan app."),
    ),
    dict(
        name="job_scam", sender="mobile", label=_scam(["too_good_to_be_true", "advance_fee", "unknown_sender"], "medium"),
        variants=[
            ("en", "Part time job! Earn Rs 3000-8000 daily by liking YouTube videos from home. No experience needed. WhatsApp HR {scammer} {phone}"),
            ("hi", "घर बैठे रोज़ Rs 5000 कमाएँ! सिर्फ़ वीडियो लाइक करने हैं। रजिस्ट्रेशन के लिए {phone} पर व्हाट्सऐप करें।"),
            ("en", "Congratulations, you are shortlisted for an Amazon work-from-home job. Pay Rs {fee} security deposit to start. Contact {phone}."),
            ("hinglish", "Ghar baithe part time kaam, roz Rs 4000 kamayein. Bas online task complete karein. Joining ke liye {phone} par message karein."),
        ],
        exp=("यह नौकरी का झूठा ऑफ़र (स्कैम) है। घर बैठे आसान काम से रोज़ हज़ारों रुपये का वादा झूठा है; बाद में ये 'टास्क' या 'डिपॉज़िट' के नाम पर पैसे जमा करवाते हैं।",
             "This is a fake job offer. Promises of thousands of rupees a day for easy tasks are false; they later make you pay 'deposits' or 'task fees'."),
        todo=("संपर्क न करें और कोई पैसा न भेजें। नंबर ब्लॉक कर दें।",
              "Do not contact them or send any money. Block the number."),
    ),
    dict(
        name="sender_mismatch_debit", sender="mobile", label=_scam(["unknown_sender", "sender_mismatch", "call_this_number"], "high"),
        variants=[
            ("en", "{bank}: Rs {amt} debited from your A/c XX{l4} on {date}. If not done by you, call {phone} immediately to cancel the transaction."),
            ("hi", "{bank}: आपके खाते XX{l4} से Rs {amt} कट गए हैं। अगर आपने नहीं किया तो तुरंत {phone} पर कॉल करके ट्रांज़ैक्शन रद्द करवाएँ।"),
            ("en", "Dear customer Rs {amt} is debited from {bank} A/c XX{l4}. Not you? Call customer care {phone} and share OTP to block the payment."),
            ("hinglish", "{bank}: A/c XX{l4} se Rs {amt} kat gaye. Aapne nahi kiya to turant {phone} par call karke cancel karayein."),
        ],
        exp=("यह संदेश बैंक का दिखता है पर असल में एक आम मोबाइल नंबर से आया है — बैंक ऐसे संदेश नहीं भेजते। दिया गया नंबर ठगों का है, जो फोन करने पर OTP लेकर सच में पैसे निकाल लेंगे।",
             "This looks like a bank alert but it came from a personal mobile number — banks do not send alerts that way. The number belongs to fraudsters who will take your OTP and actually steal money."),
        todo=("उस नंबर पर फोन न करें। अपनी पासबुक या कार्ड के पीछे लिखे बैंक के ऑफिशियल नंबर पर ख़ुद फोन करके जाँच करें।",
              "Do not call that number. Call the official number printed on your passbook or the back of your card to check."),
    ),
    dict(
        name="wrong_number_opener", sender="mobile",
        label=dict(doc_type="other", verdict="suspicious", is_dangerous=False, urgency="low", red_flags=["unknown_sender", "wrong_number_opener"]),
        variants=[
            ("en", "Hello, is this {name} ji? I got your number from a friend, I want to discuss something."),
            ("hi", "नमस्ते, क्या आप {name} जी बोल रहे हैं? आपका नंबर एक दोस्त से मिला है, कुछ बात करनी थी।"),
            ("en", "Hi, sorry to bother you, are you the owner of the plot near {place}? Please reply."),
            ("hinglish", "Hello, kya aap {name} ji hain? Aapka number ek dost ne diya tha, zaroori baat karni hai."),
        ],
        exp=("यह किसी अनजान व्यक्ति का 'ग़लत नंबर' जैसा संदेश है। कई ठगी इसी तरह बातचीत शुरू करके बाद में निवेश या पैसे की बात करती है।",
             "This is a 'wrong number' style message from a stranger. Many frauds start chatting this way and later bring up investments or money."),
        todo=("जवाब न दें और कोई निजी जानकारी न बताएं। बार-बार आए तो नंबर ब्लॉक कर दें।",
              "Do not reply or share personal details. Block the number if it keeps messaging."),
    ),
]


def _prescription(style, s, rng):
    drugs = rng.sample(DRUGS, rng.randint(2, 3))
    days = rng.choice([7, 15, 30, 60])
    lines, hi_parts, en_parts = [], [], []
    for i, (drug, dose, why_hi, why_en, food) in enumerate(drugs, 1):
        sched = "1-0-0" if drug == "Thyroxine" else rng.choice(list(SCHEDULES))
        if drug == "Thyroxine":
            food = "empty stomach"
        food_hi, food_en = FOOD[food]
        sched_hi, sched_en = SCHEDULES[sched]
        lines.append({
            "numbered": f"{i}. Tab. {drug} {dose}   {sched}   {food}   x {days} days",
            "dash": f"Tab {drug} {dose} - {sched} - {food} - {days} days",
            "compact": f"{drug} {dose} {sched} ({food}) {days}d",
            "table": f"| {drug} {dose} | {sched} | {food} | {days} days |",
        }[style])
        hi_parts.append(f"{drug} {why_hi} की दवा है — {sched_hi}, {food_hi} लेनी है।")
        en_parts.append(f"{drug} is for {why_en} — take it {sched_en}, {food_en}.")
    header = {"table": "| Medicine | Dose | When | Duration |", "compact": "Rx:"}.get(style, "Rx")
    text = f"Dr. {s['doctor']}, MBBS, MD (Medicine)\n{header}\n" + "\n".join(lines) + f"\nReview after {days} days."
    exp_hi = f"यह डॉ. {s['doctor']} का दवा का पर्चा है। " + " ".join(hi_parts) + f" {days} दिन बाद डॉक्टर को फिर दिखाना है।"
    exp_en = f"This is a prescription from Dr. {s['doctor']}. " + " ".join(en_parts) + f" See the doctor again after {days} days."
    todo_hi = "दवाइयाँ रोज़ तय समय पर लें और डॉक्टर से पूछे बिना कोई दवा बंद न करें।"
    todo_en = "Take the medicines at the same time every day and do not stop any medicine without asking the doctor."
    return text, (exp_hi, exp_en), (todo_hi, todo_en)


LABEL_KEYS = ("doc_type", "verdict", "is_dangerous", "urgency", "red_flags", "explanation", "what_to_do")


def _holdout_index(case):
    if case["name"] == "prescription":
        return case["variants"].index("compact")
    return max(i for i, (lang, _) in enumerate(case["variants"]) if lang == "en")


def make_example(case, split, rng, reply_language=None):
    s = _slots(rng)
    variants = case["variants"]
    held = _holdout_index(case)
    pool = [variants[held]] if split == "test" else [v for i, v in enumerate(variants) if i != held]
    variant = rng.choice(pool)
    if case["name"] == "prescription":
        input_lang = "en"
        text, exp, todo = _prescription(variant, s, rng)
    else:
        input_lang, template = variant
        text = template.format(**s)
        exp = tuple(e.format(**s) for e in case["exp"])
        todo = tuple(t.format(**s) for t in case["todo"])
    lang = reply_language or ("hi" if rng.random() < 0.7 else "en")
    idx = 0 if lang == "hi" else 1
    fields = dict(case["label"], explanation=exp[idx], what_to_do=todo[idx])
    label = {k: fields[k] for k in LABEL_KEYS}
    sender = _sender(case["sender"], s, rng)
    messages = build_messages(sender, text, lang)
    messages.append({"role": "assistant", "content": json.dumps(label, ensure_ascii=False)})
    return {
        "messages": messages,
        "meta": {"case": case["name"], "input_lang": input_lang, "reply_language": lang, "sender": sender, "text": text},
    }


def generate(n, split, seed):
    rng = random.Random(seed)
    seen, rows = set(), []
    while len(rows) < n:
        case = CASES[len(rows) % len(CASES)]
        row = make_example(case, split, rng)
        key = (row["meta"]["sender"], row["meta"]["text"], row["meta"]["reply_language"])
        if key not in seen:
            seen.add(key)
            rows.append(row)
    rng.shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--train", type=int, default=2400)
    ap.add_argument("--valid", type=int, default=150)
    ap.add_argument("--test", type=int, default=315)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, n, seed in [("train", args.train, 1), ("valid", args.valid, 2), ("test", args.test, 3)]:
        rows = generate(n, split, seed)
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} examples -> {out / f'{split}.jsonl'}")


if __name__ == "__main__":
    main()
