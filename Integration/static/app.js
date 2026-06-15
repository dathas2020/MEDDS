async function startWebcam(){

    const video =
        document.getElementById("webcam");

    try{

        const stream =
            await navigator.mediaDevices.getUserMedia({
                video:true,
                audio:false
            });

        video.srcObject = stream;
        document.getElementById("cameraStatus").innerHTML =
            "🟢 Camera Connected";

        document.getElementById("cameraStatus")
            .classList.add("status-green");

    }
    catch(error){

        console.error(error);

        document.getElementById("cameraStatus").innerHTML =
            "🔴 Camera Disconnected";

        document.getElementById("cameraStatus")
            .classList.add("status-red");

        alert("Unable to access webcam");

    }
}

async function updatePredictions(){

    console.log("updatePredictions called");

    const response =
        await fetch("/get_predictions");

    const data =
        await response.json();

    console.log(data);

    if(!data.face_detected){

        document.getElementById(
            "finalEmotion"
        ).innerText =
            "No Face Detected";

        document.getElementById(
            "finalConfidence"
        ).innerText =
            "-";

        document.getElementById(
            "topEmotion1"
        ).innerText =
            "-";

        document.getElementById(
            "topEmotion2"
        ).innerText =
            "-";

        document.getElementById(
            "topEmotion3"
        ).innerText =
            "-";

        document.getElementById(
            "distractionLevel"
        ).innerText =
            "No Face Detected";

        document.getElementById(
            "distractionScore"
        ).innerText =
            "100%";

        document.getElementById(
            "spoofStatus"
        ).innerText =
            "-";

        return;
    }

    document.getElementById(
        "finalEmotion"
    ).innerText =
        data.final_emotion;

    const emotionColors = {

        Happy:"#22c55e",
        Sad:"#3b82f6",
        Angry:"#ef4444",
        Fear:"#f59e0b",
        Neutral:"#94a3b8",
        Surprise:"#a855f7",
        Disgust:"#10b981",

        "No Face":"#ffffff"
    };

    document.getElementById(
        "finalEmotion"
    ).style.color =
        emotionColors[data.final_emotion] ||
        "#ffffff";

    document.getElementById(
        "finalConfidence"
    ).innerText =
        `${data.final_confidence}%`;

    document.getElementById(
        "spoofStatus"
    ).innerText =
        data.spoof_status;

    document.getElementById(
        "distractionLevel"
    ).innerText =
        data.distraction_level;

    document.getElementById(
        "distractionScore"
    ).innerText =
        `${data.distraction_score}%`;

    if(data.top3.length >= 3){

        document.getElementById(
            "topEmotion1"
        ).innerText =
            `${data.top3[0][0]}: ${data.top3[0][1]}%`;

        document.getElementById(
            "topEmotion2"
        ).innerText =
            `${data.top3[1][0]}: ${data.top3[1][1]}%`;

        document.getElementById(
            "topEmotion3"
        ).innerText =
            `${data.top3[2][0]}: ${data.top3[2][1]}%`;
    }
}

async function startMicrophone(){

    try{

        const stream =
            await navigator.mediaDevices.getUserMedia({
                audio:true
            });

        const audioContext =
            new AudioContext();

        const source =
            audioContext.createMediaStreamSource(stream);

        const analyser =
            audioContext.createAnalyser();

        analyser.fftSize = 128;

        source.connect(analyser);

        const canvas =
            document.getElementById("audioVisualizer");

        const ctx =
            canvas.getContext("2d");

        canvas.width =
            canvas.offsetWidth;

        canvas.height =
            canvas.offsetHeight;

        const bufferLength =
            analyser.frequencyBinCount;

        const dataArray =
            new Uint8Array(bufferLength);

        function draw(){

            requestAnimationFrame(draw);

            analyser.getByteFrequencyData(dataArray);

            ctx.clearRect(
                0,
                0,
                canvas.width,
                canvas.height
            );

            const barWidth =
                (canvas.width / bufferLength) * 2.5;

            let x = 0;

            for(let i=0;i<bufferLength;i++){

                const barHeight =
                    dataArray[i];

                ctx.fillStyle =
                    "#38bdf8";

                ctx.fillRect(
                    x,
                    canvas.height - barHeight,
                    barWidth,
                    barHeight
                );

                x += barWidth + 2;
            }
        }

        draw();

        document.getElementById("micStatus").innerHTML =
            "🟢 Microphone Connected";

        document.getElementById("micStatus")
            .classList.add("status-green");

    }
    catch(error){

        console.error(error);

        document.getElementById("micStatus").innerHTML =
            "🔴 Microphone Disconnected";

        document.getElementById("micStatus")
            .classList.add("status-red");

        alert("Microphone access denied");

    }
}

async function updateSystemStatus(){

    const response =
        await fetch("/system_status");

    const data =
        await response.json();

    setStatus(
        "embeddingStatus",
        true,
        "Embedding Pipeline Ready"
    );

    setStatus(
        "fusionModelStatus",
        true,
        "Fusion Model Ready"
    );

    setStatus(
        "spoofModelStatus",
        data.spoof_model,
        data.spoof_model
            ? "Spoof Model Ready"
            : "Spoof Model Missing"
    );

    document.getElementById(
        "modelsLoaded"
    ).innerHTML =
        `Models Loaded: ${data.models_loaded}/${data.total_models}`;
}

function setStatus(id, ok, label){

    const element =
        document.getElementById(id);

    element.classList.remove(
        "status-green",
        "status-red",
        "status-yellow"
    );

    if(ok){

        element.innerHTML =
            `🟢 ${label}`;

        element.classList.add(
            "status-green"
        );
    }
    else{

        element.innerHTML =
            `🔴 ${label}`;

        element.classList.add(
            "status-red"
        );
    }
}

async function sendFrame(){

    const video =
        document.getElementById("webcam");

    const canvas =
        document.getElementById("captureCanvas");

    const ctx =
        canvas.getContext("2d");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
    );

    const imageData =
        canvas.toDataURL("image/jpeg");

    const response =
        await fetch("/predict_face", {

            method:"POST",

            headers:{
                "Content-Type":
                "application/json"
            },

            body:JSON.stringify({
                image:imageData
            })
        });

    await response.json();
}

let mediaRecorder;

let audioChunks = [];

async function setupVoiceCapture() {

    const stream =
        await navigator.mediaDevices.getUserMedia({
            audio: true
        });

    mediaRecorder =
        new MediaRecorder(stream);

    mediaRecorder.ondataavailable =
        event => {

            audioChunks.push(event.data);

        };

    mediaRecorder.onstop =
        sendAudioChunk;

    startRecordingCycle();
}

function startRecordingCycle() {

    audioChunks = [];

    mediaRecorder.start();

    setTimeout(() => {

        mediaRecorder.stop();

    }, 3000);
}

async function sendAudioChunk() {

    const blob = new Blob(
        audioChunks,
        {
            type: "audio/webm"
        }
    );

    const formData =
        new FormData();

    formData.append(
        "audio",
        blob,
        "voice.webm"
    );

    try {

        console.log("Listening...");

        const response =
            await fetch(
                "/predict_voice",
                {
                    method: "POST",
                    body: formData
                }
            );

        const result =
            await response.json();

        console.log(
            "Voice chunk processed"
        );

    }

    catch(error) {

        console.error(error);

    }

    startRecordingCycle();
}

startWebcam();
startMicrophone();
setupVoiceCapture();

updatePredictions();
updateSystemStatus();

setInterval(updatePredictions,3000);
setInterval(updateSystemStatus,5000);
setInterval(sendFrame,3000);