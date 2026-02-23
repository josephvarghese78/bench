import random
import time
import AccessTokenTest as a
import requests
from datetime import datetime
import time
import config as cfg


def request_handler(user_session):
    #txt_trans_req(user_session)
    return txt_trans_req(user_session)


def tdsr(user_session):
    test_name="tdsr"
    return user_session.get("http://nl:3200/expense/tdsr", timeout=10000, verify=True), test_name

def api_request_main(user_session):
    tt = random.choice(cfg.think_time)
    start_time = None
    start_time_pc = None
    end_time = None
    end_time_pc = None
    response_time = None
    test_name=None
    try:

        time.sleep(tt)
        start_time = datetime.now()
        start_time_pc = time.perf_counter()

        resp, test_name=request_handler(user_session)

        end_time = datetime.now()
        end_time_pc = time.perf_counter()
        response_time = (end_time_pc - start_time_pc) * 1000

        try:
            resp_content = resp.json()
        except ValueError:
            resp_content = resp.text

        status_code = resp.status_code
        error_flag = 0 if resp.status_code in cfg.valid_status_codes + cfg.ignore_status_codes else 1


        status_code = resp.status_code
        error_flag = ""
        if resp.status_code in cfg.valid_status_codes:
            error_flag="P"
        elif resp.status_code in cfg.ignore_status_codes:
            error_flag="W"
        else:
            error_flag="F"

        return resp_content, status_code, error_flag, tt, test_name, start_time, start_time_pc, end_time, end_time_pc, response_time
    except Exception as e:
        return str(e), 0, 1, tt, test_name, start_time, start_time_pc, end_time, end_time_pc, response_time


def txt_trans_req(user_session):
    token = a.new_token()
    test_name = "text translation"

    tr_texts = [
        {
            "fr_lg": "en",
            "to_lg": "fr-ca",
            "text": "In the realm of imagination, where creativity knows no bounds, a world unfolds that captivates the mind and enchants the senses. This world is a tapestry woven with vibrant colors and intricate patterns, each thread telling a story of its own. As you wander through its landscapes, you encounter gardens of unspoken dreams, where flowers bloom with the hues of forgotten wishes. The air is filled with the melody of whispered secrets, carried on the gentle breeze that dances through the leaves of ancient trees. Each step taken is an exploration of wonder, leading you to places where time holds no power and possibilities are endless. The rivers here flow with the essence of curiosity, urging you to dive deeper into the mysteries that lie beneath their shimmering surfaces. In this place, the skies are painted with the colors of dawn and dusk, merging in a harmonious symphony that defies the ordinary. It is a world where imagination reigns supreme, inviting you to dream without limits and to explore beyond the horizon of your own thoughts."
        },
        {
            "fr_lg": "en",
            "to_lg": "ja",
            "text": "In today's rapidly evolving digital landscape, the integration of technology into everyday life has become both ubiquitous and essential. From the moment we wake up to the sound of an alarm on our smartphones to the time we spend checking emails, social media, and news updates, technology is interwoven into the fabric of our daily routines. As we commute to work, we often rely on GPS systems to navigate efficiently, while listening to personalized playlists curated by sophisticated algorithms. At the workplace, productivity tools and cloud-based applications facilitate seamless collaboration across teams, regardless of geographic location. Video conferencing platforms have redefined meetings, enabling face-to-face interactions without the need for physical presence. In our leisure time, streaming services provide on-demand entertainment, offering a vast array of choices tailored to individual preferences. As we embrace these innovations, it's crucial to remain mindful of the balance between technology and personal well-being, ensuring that we harness its potential responsibly."
        },
        {
            "fr_lg": "en",
            "to_lg": "de",
            "text": "Manulife, a leading international financial services group, operates primarily as John Hancock in the United States, and provides a wide range of financial protection and wealth management products and services to individual and institutional clients. With operations in Canada, Asia, and the United States, Manulife offers life insurance, long-term care services, pension products, annuities, mutual funds, and extensive banking solutions. The company's mission is to make decisions easier and lives better by providing reliable financial advice and innovative solutions tailored to meet the unique needs of their customers. Through its global team of employees and agents, Manulife is committed to conducting its business in a socially responsible manner, focusing on sustainability and environmental stewardship. The company has a strong track record of financial stability, driven by disciplined risk management and a commitment to delivering shareholder value over the long term. As they continue to expand their digital capabilities, Manulife remains focused on enhancing the customer experience and leveraging data analytics to drive personalized service."
        }
    ]

    tr_text=random.choice(tr_texts)


    payload={
          "user_object_id": "8e529c6c-cf91-4e9d-a613-6a89118ade0b",
          "user_id": "sasidan",
          "ssid": "a5889102-2a02-4311-97b5-0feec3c1f8ec",
          "country": "CAN",
          "department": "GDO-NON PROJECT",
          "segment": "Corporate & Other",
          "employeeTitle": "QA Engineer",
          "employeeType": "Y",
          "from_lang": tr_text['fr_lg'],
          "to_lang": tr_text['to_lg'],
          "model_category": "general",
          "text": tr_text['text'],
          "source": [
            "TEXT_TRANSLATION"
          ]
        }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }



    #requests.post(url, headers=headers, json=data, timeout=timeout, verify=verify)
    resp=user_session.post("https://cacd-chatmfc1-aslwp-be.azurewebsites.net/v2/translate-text",
                  headers=headers, json=payload, timeout=10000, verify=True)

    return resp, test_name

