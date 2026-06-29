from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app.config["SQLALCHEMY_DATABASE_URI"] = \
    "postgresql://postgres:postgres@localhost:5432/pat_db"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


class Patient(db.Model):
    __tablename__ = "patients"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)


class Session(db.Model):
    __tablename__ = "sessions"
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"))
    shape = db.Column(db.String(50))
    duration = db.Column(db.Float)
    points_total = db.Column(db.Integer)
    points_placed = db.Column(db.Integer)


@app.route("/sessions", methods=["POST"])
def create_session():
    data = request.json
    session = Session(
        patient_id=data["patient_id"],
        shape=data["shape"],
        duration=data["duration"],
        points_total=data["points_total"],
        points_placed=data["points_placed"]
    )
    print("Получен POST:", data)
    db.session.add(session)
    db.session.commit()

    return jsonify({"status": "saved"}), 201


@app.route("/patients/<int:patient_id>/stats", methods=["GET"])
def get_stats(patient_id):
    sessions = Session.query.filter_by(patient_id=patient_id).all()

    avg_duration = db.session.query(
        db.func.avg(Session.duration)
    ).filter_by(patient_id=patient_id).scalar()

    return jsonify({
        "total_sessions": len(sessions),
        "average_duration": avg_duration
    })

@app.route("/patients", methods=["POST"])
def create_patient():
    data = request.json
    patient = Patient(name=data["name"])
    db.session.add(patient)
    db.session.commit()
    return jsonify({
        "id": patient.id,
        "name": patient.name
    }), 201



@app.route("/patients", methods=["GET"])
def get_patients():
    patients = Patient.query.all()
    return jsonify([
        {"id": p.id, "name": p.name}
        for p in patients
    ])

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)

