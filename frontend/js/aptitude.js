// Select 10 random questions
let questions = [...questionBank]
    .sort(() => Math.random() - 0.5)
    .slice(0, 10);

let currentQuestion = 0;

let answers = new Array(10).fill(-1);

let timer = 30;

let interval;

// -----------------------------

function startTimer() {

    clearInterval(interval);

    timer = 30;

    document.getElementById("timer").innerHTML = timer;

    interval = setInterval(() => {

        timer--;

        document.getElementById("timer").innerHTML = timer;

        if (timer == 0) {

            nextQuestion();

        }

    }, 1000);

}

// -----------------------------

function loadQuestion() {

    clearInterval(interval);

    startTimer();

    document.getElementById("question-number").innerHTML =
        `Question ${currentQuestion + 1} / 10`;

    document.getElementById("question").innerHTML =
        questions[currentQuestion].question;

    let html = "";

    questions[currentQuestion].options.forEach((option, index) => {

        let checked = "";

        if (answers[currentQuestion] == index)
            checked = "checked";

        html += `

        <div class="option">

            <label>

            <input
                type="radio"
                name="option"
                value="${index}"
                ${checked}
                onchange="saveAnswer(${index})"
            >

            ${option}

            </label>

        </div>

        `;

    });

    document.getElementById("options").innerHTML = html;

}

// -----------------------------

function saveAnswer(index) {

    answers[currentQuestion] = index;

}

// -----------------------------

function nextQuestion() {

    if (currentQuestion < 9) {

        currentQuestion++;

        loadQuestion();

    }

    else {

        finishTest();

    }

}

// -----------------------------

function previousQuestion() {

    if (currentQuestion > 0) {

        currentQuestion--;

        loadQuestion();

    }

}

// -----------------------------

function finishTest() {

    clearInterval(interval);

    let score = 0;

    questions.forEach((q, index) => {

        if (answers[index] == q.answer)

            score++;

    });

    let aptitudeScore = score * 10;

    localStorage.setItem(
        "aptitudeScore",
        aptitudeScore
    );

    alert(

        `Your Aptitude Score is ${aptitudeScore}/100`

    );

    window.location.href = "profile.html";

}

// -----------------------------

loadQuestion();