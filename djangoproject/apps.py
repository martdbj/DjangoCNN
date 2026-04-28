from django.apps import AppConfig


class AppConfig(AppConfig):
    name = 'djangoproject'
    predictor = None

    def ready(self):
        from .ai.cnn_init import AiInitializer
        if not AppConfig.predictor:
            AppConfig.predictor = AiInitializer()