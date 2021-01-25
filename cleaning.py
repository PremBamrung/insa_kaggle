import string
import re
from itertools import groupby

PUNCT_TO_REMOVE = string.punctuation


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


def remove_urls(text):
    url_pattern = re.compile(
        r'https?://\S+|www\.\S+|[\S]+\.(net|com|org|info|edu|gov|uk|de|ca|jp|fr|au|us|ru|ch|it|nel|se|no|es|mil)[\S]*\s?')
    return url_pattern.sub(r'website ', text)


def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)


def remove_extra_spaces(text):
    space_pattern = r'\s+|^\s+|\s+$'
    without_space = re.sub(pattern=space_pattern, repl=" ", string=text)
    return without_space


def remove_email(text):
    email_pattern = r'\S*@\S*\s?'
    without_email = re.sub(pattern=email_pattern, repl=" email ", string=text)
    return without_email


def remove_phone(text):
    phone_pattern = re.compile(
        ".*?(\(?\d{3}\D{0,3}\d{3}\D{0,3}\d{4}).*?", re.S)
    without_phone = re.sub(pattern=phone_pattern,
                           repl=" phone number ", string=text)
    return without_phone


def remove_non_letters(txt):
    return re.sub('[^a-z_]', ' ', txt)


def remove_number(text):
    without_number = ' '.join(s for s in text.split()
                              if not any(c.isdigit() for c in s))
    return without_number


def remove_parenthese(text):
    return re.sub(r'\([^)]*\)', '', text)


def remove_special(text):
    # return re.sub('[@#$=+}~"()-|`_^{]', '', text)
    return text.translate(str.maketrans('', '', '[@#$]=+}~"()-|`_^{/*+'))


def remove_duplicated(text):
    return " ".join([k for k, v in groupby(text.split())])


def clean(text):
    text = remove_urls(text)
    text = remove_email(text)
    text = remove_html(text)
    text = remove_parenthese(text)
    # text = remove_phone(text)
    text = remove_number(text)
    text = remove_special(text)
    text = remove_duplicated(text)
    text = remove_extra_spaces(text)

    return text


if __name__ == '__main__':
    text = "  On this website https://pypi.org/sponsor/. I really like,  [ ] ^ carotte banana user@xxx.com 123 any@www (740) 522-6300   78@ppp @5555 aa@111   000-000-0000  Without Mr Modi they are BIG ZEROS 45-45-145sd-  45gre6  "
    print(text)
    print('')
    print(clean(text))
