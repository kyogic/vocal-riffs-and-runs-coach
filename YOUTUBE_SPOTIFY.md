# YouTube and Spotify Integration Guide

## YouTube Support ✅

The Vocal Coach now supports downloading audio directly from YouTube! This is the easiest way to practice with your favorite songs.

### How to Use YouTube Integration

1. **Copy a YouTube URL**
   - Go to YouTube and find the song you want to practice
   - Copy the URL from your browser
   - Supported formats:
     - `https://www.youtube.com/watch?v=VIDEO_ID`
     - `https://youtu.be/VIDEO_ID`

2. **Paste into the app**
   - Open Vocal Coach
   - Paste the URL into the "Or YouTube URL:" field
   - Click "🎵 Download from YouTube"

3. **Wait for download**
   - The app will:
     - Connect to YouTube
     - Fetch video information
     - Download the best quality audio
     - Convert to WAV format
     - Load it automatically

4. **Start analyzing**
   - Once loaded, click "🔍 Analyze Pitch"
   - Practice with the visualizations!

### Requirements for YouTube

You need **FFmpeg** installed for YouTube downloads to work:

**Windows (using Chocolatey):**
```powershell
choco install ffmpeg
```

**Windows (manual):**
1. Download from: https://www.ffmpeg.org/download.html
2. Extract the files
3. Add the `bin` folder to your system PATH

**Verify FFmpeg is installed:**
```powershell
ffmpeg -version
```

### Downloaded Files Location

YouTube audio files are saved to:
- Windows: `C:\Users\YourName\.vocal_coach\downloads\`
- The app automatically creates this folder

Files are saved with the video title as the filename and converted to WAV format for best compatibility.

### YouTube Download Tips

1. **Choose high-quality audio**
   - The app automatically downloads the best available audio quality
   - Music videos usually have better audio than live performances

2. **Be patient with long videos**
   - Longer songs take more time to download
   - The progress label shows what's happening

3. **Internet connection required**
   - Downloads require an active internet connection
   - Download once, then use offline

4. **Troubleshooting downloads**
   - If download fails, check your internet connection
   - Make sure FFmpeg is installed
   - Try a different video URL
   - Some videos may be region-locked or restricted

### Legal Considerations

- Only download content you have permission to access
- YouTube downloads are for personal use only
- Respect copyright and artist rights
- Don't redistribute downloaded content

## Spotify Support ❌ (Not Available)

Unfortunately, **Spotify integration is not currently available** due to technical and legal limitations.

### Why No Spotify?

1. **API Restrictions**
   - Spotify's API does not provide direct access to audio streams
   - Only 30-second previews are available via API
   - Full playback requires Spotify Premium and their SDK

2. **Legal Issues**
   - Extracting full audio from Spotify violates their Terms of Service
   - Could result in account termination
   - Potential copyright infringement

3. **Technical Limitations**
   - Spotify uses encrypted streaming
   - Audio is DRM-protected
   - No legitimate way to access for pitch analysis

### Alternatives to Spotify

If you want to practice with Spotify songs:

#### Option 1: Use YouTube
- Most Spotify songs are also on YouTube
- Search for "[Song Name] [Artist]" on YouTube
- Use the YouTube integration in Vocal Coach

#### Option 2: Local Files
- If you own the music (purchased downloads)
- Upload your files to the app using "📁 Load Audio File"
- Supports MP3, WAV, FLAC formats

#### Option 3: Spotify + YouTube Workflow
1. Find songs you like on Spotify
2. Note the song name and artist
3. Search for the same song on YouTube
4. Download via Vocal Coach's YouTube integration

#### Option 4: Record from Spotify (Advanced)
If you have Spotify Premium:
1. Use audio recording software (e.g., Audacity)
2. Record system audio while playing on Spotify
3. Save as WAV or MP3
4. Load into Vocal Coach

**Note:** Recording should only be for personal practice purposes.

## Future Audio Sources

We're exploring these options for future releases:

### Possible Future Integrations

1. **SoundCloud**
   - More permissive API
   - Many independent artists
   - Good for discovering new music

2. **Apple Music / iTunes**
   - If user has local iTunes library
   - Read from local files only

3. **Bandcamp**
   - Artist-friendly platform
   - Many offer streaming or downloads

4. **Local Music Library Integration**
   - Scan your computer for audio files
   - Build a library within the app
   - Quick access to your collection

## Best Practices

### For Best Results

1. **Start with YouTube**
   - Easiest option
   - Works for most popular songs
   - High quality audio

2. **Use Official Audio**
   - Look for "Official Audio" or "Official Video"
   - Better quality than live performances
   - Clearer vocals for pitch detection

3. **Check Audio Quality**
   - Higher quality = better pitch detection
   - Studio versions work best
   - Avoid heavily compressed audio

4. **Build Your Practice Library**
   - Download songs you practice regularly
   - Keeps them available offline
   - Faster to load than re-downloading

## Troubleshooting

### "yt-dlp not installed" Error

**Solution:**
```powershell
pip install yt-dlp
```

### "FFmpeg not found" Error

**Solution:**
1. Install FFmpeg (see requirements above)
2. Make sure it's in your system PATH
3. Restart the application

### Download is Very Slow

**Solutions:**
- Check your internet speed
- Try a different time of day
- Download shorter songs first to test

### "Video Unavailable" Error

**Possible causes:**
- Video is private or deleted
- Region restrictions
- Age restrictions
- Network issues

**Solutions:**
- Try a different video
- Use a VPN (if regional)
- Check the URL is correct

### Downloaded File Won't Load

**Solutions:**
- Check if FFmpeg conversion worked
- Try downloading again
- Use a different video
- Load a local file instead

## Summary

| Feature | Status | Notes |
|---------|--------|-------|
| YouTube | ✅ Available | Requires FFmpeg |
| Spotify | ❌ Not Available | Use YouTube instead |
| SoundCloud | 🔮 Future | Under consideration |
| Local Files | ✅ Available | MP3, WAV, FLAC |

**Recommended:** Use YouTube integration for the best experience with online content!
