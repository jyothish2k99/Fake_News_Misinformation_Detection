import asyncio
import io
import socket
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app import main


class UrlSafetyTests(unittest.TestCase):
    def test_loopback_url_is_rejected(self):
        with self.assertRaises(HTTPException) as error:
            main._validate_public_url("http://127.0.0.1:8000/api/health")

        self.assertEqual(error.exception.status_code, 400)

    @patch("app.main.socket.getaddrinfo")
    def test_public_dns_address_is_allowed(self, getaddrinfo):
        getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ]

        main._validate_public_url("https://example.com/article")

    @patch("app.main.socket.getaddrinfo")
    def test_mixed_public_and_private_dns_answers_are_rejected(self, getaddrinfo):
        getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.2", 443)),
        ]

        with self.assertRaises(HTTPException):
            main._validate_public_url("https://example.com/article")

    def test_redirect_destination_is_validated_before_second_request(self):
        response = Mock(status_code=302, headers={"Location": "http://127.0.0.1/private"})
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        session = Mock()
        session.get.return_value = response

        def validate(url):
            if "127.0.0.1" in url:
                raise HTTPException(status_code=400, detail="blocked")

        with patch("app.main.requests.Session", return_value=session), patch.object(main, "_validate_public_url", side_effect=validate):
            with self.assertRaises(HTTPException):
                main._fetch_article("https://example.com/article")

        self.assertEqual(session.get.call_count, 1)

    def test_response_content_length_is_bounded(self):
        response = Mock(status_code=200, headers={"Content-Length": str(main.MAX_EXTRACT_BYTES + 1)})
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        session = Mock()
        session.get.return_value = response

        with patch("app.main.requests.Session", return_value=session), patch.object(main, "_validate_public_url"):
            with self.assertRaises(HTTPException) as error:
                main._fetch_article("https://example.com/article")

        self.assertEqual(error.exception.status_code, 413)


class BatchWorkflowTests(unittest.TestCase):
    def test_empty_csv_is_rejected(self):
        upload = UploadFile(filename="articles.csv", file=io.BytesIO(b"title,text\n"))

        with self.assertRaises(HTTPException) as error:
            asyncio.run(main.batch(upload))

        self.assertEqual(error.exception.status_code, 400)

    def test_batch_keeps_existing_csv_report_columns(self):
        content = b"title,text\nHeadline,An article body long enough for a test prediction.\n"
        upload = UploadFile(filename="articles.csv", file=io.BytesIO(content))
        prediction = {
            "label": "FAKE-LIKE",
            "fake_probability": 0.8,
            "real_probability": 0.2,
            "confidence": 0.8,
        }

        with patch.object(main.model, "pipeline", object()), patch.object(main.model, "predict", return_value=prediction):
            response = asyncio.run(main.batch(upload))
            body = asyncio.run(self.read_response(response))

        self.assertIn(b"title,text,prediction,fake_probability,real_probability,confidence", body)
        self.assertIn(b"FAKE-LIKE", body)

    async def read_response(self, response):
        return b"".join([chunk async for chunk in response.body_iterator])


class PredictionIndependenceTests(unittest.TestCase):
    def test_source_score_does_not_change_prediction(self):
        model_result = {
            "label": "REAL-LIKE",
            "fake_probability": 0.1,
            "real_probability": 0.9,
            "confidence": 0.9,
            "explanation": [],
            "disclaimer": "Model estimate only.",
        }
        request = main.ArticleRequest(title="A test headline", text="This article body contains enough characters to meet input validation.")

        with patch.object(main.model, "predict", return_value=model_result.copy()), patch.object(main.evidence, "search", return_value=[]), patch("app.main.score_for_url", side_effect=[{"available": True, "credibility_score": 99.0}, {"available": False, "message": "unavailable"}]):
            high_source = main.predict(request)
            unknown_source = main.predict(request)

        for key in ("label", "fake_probability", "real_probability", "confidence"):
            self.assertEqual(high_source[key], unknown_source[key])


if __name__ == "__main__":
    unittest.main()