import argparse
import os
from datetime import datetime

import autorootcwd
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

SLACK_WEBHOOK_URL = os.getenv("SERVER_SLACK_URL")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Send a Slack message with repo and custom message."
    )
    parser.add_argument("--repo", required=True, help="Repository name")
    parser.add_argument(
        "--message", required=False, default="Done", help="Message to send"
    )
    return parser.parse_args()


def send_slack_message(attachments):
    response = requests.post(SLACK_WEBHOOK_URL, json={"attachments": attachments})
    if response.status_code != 200:
        raise ValueError(
            f"Request to Slack returned an error {response.status_code}, the response is: {response.text}"
        )


def alert(repo, message):
    send_slack_message(
        [
            {
                "fallback": f"{repo} - {message}",
                "color": "#FFA500",
                "title": "코드 수행",
                "text": f"{repo} - {message} 완료",
                "ts": datetime.now().timestamp(),
            }
        ]
    )


if __name__ == "__main__":
    args = parse_arguments()
    send_slack_message(
        [
            {
                "fallback": f"{args.repo} - {args.message}",
                "color": "#FFA500",
                "title": "코드 수행",
                "text": f"{args.repo} - {args.message} 완료",
                "ts": datetime.now().timestamp(),
            }
        ]
    )
    print("코드 수행 완료")
