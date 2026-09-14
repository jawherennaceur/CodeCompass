"""Petit module d'exemple pour tester le chunking."""


def connect_db(host, port):
    """Établit une connexion à la base de données."""
    print(f"Connexion à {host}:{port}")
    return {"host": host, "port": port, "connected": True}


def retry_api_call(func, max_retries=3):
    """Réessaie un appel API en cas d'échec."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            print(f"Tentative {attempt + 1} échouée: {e}")
    raise RuntimeError("Toutes les tentatives ont échoué")


class UserAuth:
    """Gère l'authentification des utilisateurs."""

    def __init__(self, secret_key):
        self.secret_key = secret_key

    def login(self, username, password):
        """Vérifie les identifiants et retourne un token."""
        if self._check_credentials(username, password):
            return self._generate_token(username)
        return None

    def _check_credentials(self, username, password):
        return len(password) > 0

    def _generate_token(self, username):
        return f"token_{username}"