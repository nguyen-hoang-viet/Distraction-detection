function uploadVideo() {
    var input = document.getElementById("videoUpload");
    var file = input.files[0]; // Lấy tệp được chọn
    var videoPlayer = document.getElementById("videoPlayer");
    var videoContainer = document.getElementById("videoContainer");

    if (file) {
        videoPlayer.src = URL.createObjectURL(file);
        videoContainer.style.display = "block"; // Hiển thị video container
        alert("Tệp video đã được chọn: " + file.name);
    } else {
        alert("Vui lòng chọn một tệp video.");
    }
}

function deleteVideo() {
    var videoPlayer = document.getElementById("videoPlayer");
    var videoContainer = document.getElementById("videoContainer");

    videoPlayer.src = ""; // Xóa nguồn video
}

function displayNotification() {
    var currentTime = new Date();
    var formattedTime = currentTime.toLocaleTimeString();
    var notificationBox = document.getElementById("notificationBox");
    notificationBox.innerHTML += "Thời gian hiện tại là: " + formattedTime + "-------------------------------" + "\n";
}

// Hiển thị thông báo mỗi 3 giây
setInterval(displayNotification, 3000);

// Hàm tải toàn bộ thông báo về máy
function downloadNotifications() {
    var notifications = document.getElementById("notificationBox").innerHTML;
    var blob = new Blob([notifications], { type: "text/plain" });
    var url = URL.createObjectURL(blob);

    var a = document.createElement("a");
    a.href = url;
    a.download = "notifications.txt";
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();

    // Thu hồi URL đã tạo ra sau khi tải xong
    setTimeout(function () {
        window.URL.revokeObjectURL(url);
    }, 100);
}