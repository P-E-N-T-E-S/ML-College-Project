# JupyterHub Configuration File
import os

# Network configuration
c.JupyterHub.ip = '0.0.0.0'
c.JupyterHub.port = 8000
c.JupyterHub.hub_ip = '0.0.0.0'
c.JupyterHub.hub_connect_ip = '127.0.0.1'

# Authentication - Simple (no password for development)
c.JupyterHub.authenticator_class = 'dummy'

# Allow any username
c.Authenticator.admin_users = {'admin'}
c.Authenticator.allow_all = True

# Spawner configuration
c.JupyterHub.spawner_class = 'jupyterhub.spawner.SimpleLocalProcessSpawner'

# Increase timeout for spawner
c.Spawner.start_timeout = 120
c.Spawner.http_timeout = 120

# Set the default URL to JupyterLab
c.Spawner.default_url = '/lab'

# Working directory for users
c.Spawner.notebook_dir = '/home/{username}/work'

# Command to start single-user server
c.Spawner.cmd = ['jupyter-labhub']

# Environment variables for spawned notebooks
c.Spawner.environment = {
    'JUPYTER_ENABLE_LAB': 'yes',
    'JUPYTERHUB_SINGLEUSER_APP': 'jupyter_server.serverapp.ServerApp',
}

# Args for the notebook server
c.Spawner.args = ['--allow-root']

# Disable SSL (handled by reverse proxy if needed)
c.JupyterHub.ssl_key = ''
c.JupyterHub.ssl_cert = ''

# Logging
c.JupyterHub.log_level = 'DEBUG'
c.Application.log_level = 'DEBUG'
c.Spawner.debug = True
