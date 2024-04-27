from abc import ABC, abstractmethod


class BaseParserPDF(ABC):

    @abstractmethod
    async def get_text_from_pdf(self, file_path: str):
        pass
