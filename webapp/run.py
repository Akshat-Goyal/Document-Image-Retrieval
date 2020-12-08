from webapp import app
from webapp.app.route import routes

app.register_blueprint(routes)

app.run(debug=True)
