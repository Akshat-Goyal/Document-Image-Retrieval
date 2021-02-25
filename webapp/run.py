from webapp import app
from webapp.app.route import routes

app.register_blueprint(routes)

app.run(host="0.0.0.0", debug=True)
