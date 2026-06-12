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

    const response =
        await fetch("/get_predictions");

    const data =
        await response.json();

    document.getElementById("faceEmotion").innerText =
        `${data.face_emotion} (${data.face_confidence}%)`;

    document.getElementById("voiceEmotion").innerText =
        `${data.voice_emotion} (${data.voice_confidence}%)`;

    document.getElementById("spoofStatus").innerText =
        `${data.spoof_status} (${data.spoof_confidence}%)`;

    document.getElementById("finalEmotion").innerText =
        data.final_emotion;
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
        "faceModelStatus",
        data.face_model,
        data.face_model
            ? "Face Model Ready"
            : "Face Model Missing"
    );

    setStatus(
        "voiceModelStatus",
        data.voice_model,
        data.voice_model
            ? "Voice Model Ready"
            : "Voice Model Missing"
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

    const result =
        await response.json();

    console.log(result);
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

        document.getElementById(
            "voiceEmotion"
        ).innerText =
            "Listening...";

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

        updateVoiceCard(result);

    }

    catch(error) {

        console.error(error);

    }

    startRecordingCycle();
}

function updateVoiceCard(result) {

    if(!result.success)
        return;

    document.getElementById(
        "voiceEmotion"
    ).innerText =
        result.emotion;

    document.getElementById(
        "voiceConfidence"
    ).innerText =
        result.confidence + "%";
}

startWebcam();
startMicrophone();
setupVoiceCapture();

updatePredictions();
updateSystemStatus();

setInterval(updatePredictions,3000);
setInterval(updateSystemStatus,5000);
setInterval(sendFrame,3000);