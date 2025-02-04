import logging
import os
import telebot
from telebot import types
from PIL import Image
import io
from typing import List
import hashlib
import ML_functions
import numpy as np

BOT_TOKEN = "Your Token"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

imgs = {}  # {user_id: [img1, img2, ...]} (Now stores images as numpy arrays)
current_state = {}  # {user_id: "start" or "identifying"}
last_message_ids = {}  # {user_id: message_id}

bot = telebot.TeleBot(BOT_TOKEN)

def send_start_message(chat_id, text):
    markup = types.InlineKeyboardMarkup(row_width=1)
    item = types.InlineKeyboardButton("Перейти к идентификации", callback_data='identify')
    markup.add(item)

    message = bot.send_message(chat_id, text, reply_markup=markup)
    return message.message_id

def send_identify_message(chat_id, text):
    markup = types.InlineKeyboardMarkup(row_width=1)
    item = types.InlineKeyboardButton("Загрузить новую компанию", callback_data='restart')
    markup.add(item)

    message = bot.send_message(chat_id, text, reply_markup=markup)
    return message.message_id

@bot.message_handler(commands=['start'])
def start(message):
    user_id = message.from_user.id
    imgs[user_id] = []
    current_state[user_id] = "start"

    text = "Выберите компанию и загружайте её логотипы. Я буду их запоминать и далее на основе этих изображений смогу сказать про любой логотип, является ли он логотипом вашей компании. Как только вы посчитаете, что загрузили достаточно изображений, нажимайте на кнопку. "

    if last_message_ids.get(user_id):
        try:
            bot.delete_message(message.chat.id, last_message_ids[user_id])
        except telebot.apihelper.ApiTelegramException as e:
            logger.warning(f"Error deleting message: {e}")
    last_message_ids[user_id] = send_start_message(message.chat.id, text)

@bot.callback_query_handler(func=lambda call: True)
def callback_query(call):
    user_id = call.from_user.id
    if call.data == 'identify':
        if not imgs.get(user_id) or len(imgs[user_id]) == 0:
            text = "Вы еще не загрузили ни одной фотографии, рано переходить к следующему этапу."
            if last_message_ids.get(user_id):
                try:
                    bot.delete_message(call.message.chat.id, last_message_ids[user_id])
                except telebot.apihelper.ApiTelegramException as e:
                    logger.warning(f"Error deleting message: {e}")
            last_message_ids[user_id] = send_start_message(call.message.chat.id, text)
        else:
            current_state[user_id] = "identifying"
            text = "Теперь загружайте картинки про которые вы хотите узнать, принадлежат ли они данной компании. Если вы захотите сменить вашу компанию, с которой мы сверяем нажмите кнопку. "
            if last_message_ids.get(user_id):
                try:
                    bot.delete_message(call.message.chat.id, last_message_ids[user_id])
                except telebot.apihelper.ApiTelegramException as e:
                    logger.warning(f"Error deleting message: {e}")
            last_message_ids[user_id] = send_identify_message(call.message.chat.id, text)

    elif call.data == 'restart':
        new_message = telebot.types.Message(message_id=call.message.message_id,
                                          from_user=call.from_user,
                                          chat=call.message.chat,
                                          date=call.message.date,
                                          content_type='text')

        start(new_message)


@bot.message_handler(content_types=['photo'])
def image_handler(message):
    user_id = message.from_user.id
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    file = bot.download_file(file_info.file_path)
    image = Image.open(io.BytesIO(file))

    image_np = np.array(image)
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    elif image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)

    if current_state.get(user_id) == "start":
        imgs[user_id].append(image_np)
        bot.send_message(message.chat.id, "Логотип добавлен.")
        if last_message_ids.get(user_id):
            try:
                bot.delete_message(message.chat.id, last_message_ids[user_id])
            except telebot.apihelper.ApiTelegramException as e:
                logger.warning(f"Error deleting message: {e}")
        text = "Выберите компанию и загружайте её логотипы. Я буду их запоминать и далее на основе этих изображений смогу сказать про любой логотип, является ли он логотипом вашей компании. Как только вы посчитаете, что загрузили достаточно изображений, нажимайте на кнопку. "
        last_message_ids[user_id] = send_start_message(message.chat.id, text)

    elif current_state.get(user_id) == "identifying":
        imgs[user_id].append(image_np)
        for i in range(len(imgs[user_id])):
            try:
                imgs[user_id][i] = ML_functions.resize_with_padding(imgs[user_id][i], (224, 224))
                imgs[user_id][i] = ML_functions.transform(imgs[user_id][i])
            except:
                pass
        embeddings = ML_functions.get_embeddings(imgs[user_id])
        is_company = ML_functions.decide(embeddings, imgs[user_id])
        print(is_company)
        if is_company:
            bot.send_message(message.chat.id, "Это действительно логотип вашей компании")
        else:
            bot.send_message(message.chat.id, "Это не логотип вашей компании")
        imgs[user_id].pop()

        if last_message_ids.get(user_id):
            try:
                bot.delete_message(message.chat.id, last_message_ids[user_id])
            except telebot.apihelper.ApiTelegramException as e:
                logger.warning(f"Error deleting message: {e}")
        text = "Теперь загружайте картинки про которые вы хотите узнать, принадлежат ли они данной компании. Если вы захотите сменить вашу компанию, с которой мы сверяем нажмите кнопку. "
        last_message_ids[user_id] = send_identify_message(message.chat.id, text)

if __name__ == '__main__':
    try:
        logger.info("Bot started")
        bot.polling(none_stop=True)
    except KeyboardInterrupt:
        logger.info("Bot stopped")
    except Exception as e:
        logger.exception("An error occurred during polling:")
