class AppError(Exception):
    status_code = 500
    error_code = "INTERNAL_SERVER_ERROR"
    message = "unexpected error occured"

    def __init__(self, message: str | None = None):
        if message:
            self.message = message


class InvalidRequestError(AppError):
    status_code = 400
    error_code = "INVALID_REQUEST"
    message = "given request is not valid"


class EmptyStringError(AppError):
    status_code = 400
    error_code = "EMPTY_STRING"
    message = "given string is not valid"


class UnsupportedMediaTypeError(AppError):
    status_code = 415
    error_code = "UNSUPPORTED_FILE_TYPE"
    message = "only pdf files are supported"


class UnprocessableContentError(AppError):
    status_code = 422
    error_code = "UNPROCESSABLE_CONTENT"
    message = "file is corrupt"