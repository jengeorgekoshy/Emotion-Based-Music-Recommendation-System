const main = document.getElementById('main');
var flag = 1;

function songEmotion(emotion) {
  fetch('../static/predicted_songs.json')
    .then((response) => response.json())
    .then(function (data) {
      main.innerHTML = '';
      data.forEach((song) => {
        if (emotion === 'all') {
          document.getElementById('main').scrollIntoView();
          const songEl = document.createElement('div');
          songEl.classList.add('song');
          songEl.innerHTML = `
              <a href="${song.Image}" target="_blank">

              <img src="${song.Ext_link}"  alt="name">
              </a>
              <div class="Song-info">
              <h4>Song Name : "${song.SongName}" </h4>
              </div>
              <div class="other-info">
                <h4>Artist : <span>${song.Artist}</span></h4>
                <h4>Album : <span>${song.Album}</span></h4>
              </div>
              </div>
          `;

          // Append newyly created card element to the container
          main.appendChild(songEl);

          // Append newyly created card element to the container
        } else {
          if (song.Emotion === emotion && flag === 1) {
            document.getElementById('main').scrollIntoView();
            const songEl = document.createElement('div');
            songEl.classList.add('song');
            songEl.innerHTML = `
              <a href="${song.Image}" target="_blank">

              <img src="${song.Ext_link}"  alt="name">
              </a>
              <div class="Song-info">
              <h4>Song Name : "${song.SongName}" </h4>
              </div>
              <div class="other-info">
                <h4>Artist : <span>${song.Artist}</span></h4>
                <h4>Album : <span>${song.Album}</span></h4>
              </div>
              </div>
          `;

            // Append newyly created card element to the container
            main.appendChild(songEl);
          }
        }
      });
      flag = 0;
    });
}

function showAllSongs() {
  window.location.reload = true;
  songEmotion('all');
}

function toggleMenu() {
  const menuToggle = document.querySelector('.toggle');
  const nav = document.querySelector('.nav');
  menuToggle.classList.toggle('active');
  nav.classList.toggle('active');
}

function topFunction() {
  document.body.scrollTop = 0;
  document.documentElement.scrollTop = 0;
}
