export const segregate = (URL) => {
  const parts = URL.split("/");
  const filename = parts[parts.length - 1];

  let splitFilename;
  if (filename.includes(" - ")) {
    splitFilename = filename.split(" - ");
  } else if (filename.includes("- ")) {
    splitFilename = filename.split("- ");
  } else {
    throw new Error("Unexpected filename format");
  }

  const title = splitFilename[0];
  const content = splitFilename[1].split(".")[0];
  return { title, content, URL };
};

export const fetchData = async (episodeNumber) => {
  let episodeName = "";
  let nextEpisodeName = "";

  if (episodeNumber < 10) {
    episodeName = `Ep 0${episodeNumber}`;
    nextEpisodeName = `Ep 0${episodeNumber + 1}`;
  } else {
    episodeName = `Ep ${episodeNumber}`;
    nextEpisodeName = `Ep ${episodeNumber + 1}`;
  }

  try {
    const response = await fetch('https://api.github.com/repos/hack-boi/Chanakya/contents');
    const data = await response.json();
    const audioFiles = data
      .filter(file => file.name.endsWith(".mp3") || file.name.endsWith(".wav") || file.name.endsWith(".m4a") || file.name.endsWith(".aac"))
      .map(file => file.download_url);

    const URL = audioFiles.find(URL => URL.includes(episodeName));
    const nextURL = audioFiles.find(URL => URL.includes(nextEpisodeName));

    if (!URL) {
      throw new Error(`${episodeName} not found`);
    }

    let trimData = segregate(URL);
    let nextTrimData = nextURL ? segregate(nextURL) : { title: "No more episodes available,", content: "this is the finale", URL: null };

    return { trimData, nextTrimData };
  } catch (error) {
    console.error("Error fetching episode:", error);
    throw error; // Rethrow the error to propagate it upwards
  }
};

export default fetchData;
