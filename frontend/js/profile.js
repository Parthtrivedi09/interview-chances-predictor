const aptitude = localStorage.getItem("aptitudeScore");

document.getElementById("aptitudeScore").innerHTML = aptitude;

const form = document.getElementById("profileForm");

form.addEventListener("submit", async function(e){

    e.preventDefault();

    const result=document.getElementById("result");

    result.style.display="block";

    result.innerHTML=

    `<div class="loader"></div>
    <p style="text-align:center;">Analyzing Candidate...</p>`;

    const formData=new FormData(form);

    // Automatically add aptitude score
    formData.append("aptitude",aptitude);

    try{

        const response=await fetch(
            "http://127.0.0.1:8000/predict",
            {
                method:"POST",
                body:formData
            }
        );

        const data=await response.json();

        result.innerHTML=

        `
        <h2>📊 Prediction Result</h2>

        <p>

        <b>Interview Success Probability :</b>

        ${data.probability}%

        </p>

        <div class="progress">

        <div class="progress-fill"
        style="width:${data.probability}%">

        ${data.probability}%

        </div>

        </div>

        <p>

        <b>Voice Score :</b>

        ${data.voice_score}

        </p>

        <h3>

        ✅ Strengths

        </h3>

        ${(data.strengths || [])

        .map(x=>`<span class="tag strength">${x}</span>`)

        .join("")}

        <h3>

        ⚠ Weaknesses

        </h3>

        ${(data.weaknesses || [])

        .map(x=>`<span class="tag weakness">${x}</span>`)

        .join("")}

        <h3>

        💡 Suggestions

        </h3>

        ${(data.suggestions || [])

        .map(x=>`<span class="tag suggestion">${x}</span>`)

        .join("")}

        `;

    }

    catch(err){

        console.log(err);

        result.innerHTML=

        "<h2>❌ Unable to connect to backend.</h2>";

    }

});