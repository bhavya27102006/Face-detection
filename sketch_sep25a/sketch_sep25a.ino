#define GREEN_LED D0
#define RED_LED   D1

void setup() {
  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);

  digitalWrite(GREEN_LED, LOW);
  digitalWrite(RED_LED, LOW);

  Serial.begin(115200);
}

void loop() {

  if (Serial.available()) {

    char command = Serial.read();

    if (command == 'B') {
      // Bhavya detected
      digitalWrite(GREEN_LED, HIGH);
      digitalWrite(RED_LED, LOW);
    }

    else if (command == 'U') {
      // Unknown person
      digitalWrite(GREEN_LED, LOW);
      digitalWrite(RED_LED, HIGH);
    }

    else if (command == 'O') {
      // No person / idle
      digitalWrite(GREEN_LED, LOW);
      digitalWrite(RED_LED, LOW);
    }
  }
}