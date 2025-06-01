import gradio as gr
import os
import shutil
from datetime import datetime

# Constants
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB in bytes
UPLOAD_DIR = "data"  # Directory to store uploaded files

def validate_file_size(file_path):
    """Check if file size is within the limit."""
    try:
        size = os.path.getsize(file_path)
        return size <= MAX_FILE_SIZE, size
    except Exception:
        return False, 0

def format_size(size_bytes):
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"

def upload_file(files):
    """
    Process uploaded files with status updates.
    
    Args:
        files (list): List of uploaded file objects
    Returns:
        str: Status message about the upload process
    """
    if not files:
        return "⚠️ No files selected for upload."
    
    # Create upload directory if it doesn't exist
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    status_messages = []
    successful_uploads = []
    failed_uploads = []
    
    for file in files:
        try:
            if not file.name:
                failed_uploads.append("Unnamed file (Invalid file)")
                continue
            
            filename = os.path.basename(file.name)
            file_size = os.path.getsize(file.name)
            
            # Validate file size
            is_valid_size, actual_size = validate_file_size(file.name)
            if not is_valid_size:
                failed_uploads.append(f"{filename} (Size: {format_size(actual_size)} - Exceeds 100MB limit)")
                continue
            
            # Generate destination path
            dest_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.exists(dest_path):
                # Add timestamp to filename if it already exists
                name, ext = os.path.splitext(filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{name}_{timestamp}{ext}"
                dest_path = os.path.join(UPLOAD_DIR, filename)
            
            # Copy file
            shutil.copy2(file.name, dest_path)
            
            # Verify upload
            if os.path.exists(dest_path) and os.path.getsize(dest_path) == file_size:
                successful_uploads.append(f"{filename} ({format_size(file_size)})")
            else:
                failed_uploads.append(f"{filename} (Verification failed)")
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                    
        except Exception as e:
            failed_uploads.append(f"{filename if 'filename' in locals() else 'Unknown file'} (Error: {str(e)})")
    
    # Prepare status message
    if successful_uploads:
        status_messages.append(f"✅ Successfully uploaded {len(successful_uploads)} files:")
        for upload in successful_uploads:
            status_messages.append(f"   • {upload}")
    
    if failed_uploads:
        status_messages.append(f"\n❌ Failed to upload {len(failed_uploads)} files:")
        for failure in failed_uploads:
            status_messages.append(f"   • {failure}")
    
    if not status_messages:
        return "⚠️ No files were processed."
    
    return "\n".join(status_messages)

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# File Upload Test")
    gr.Markdown(f"Maximum file size: 100MB")
    
    with gr.Row():
        with gr.Column(scale=4):
            # File upload component
            upload_button = gr.UploadButton(
                "Click to Upload Files",
                file_count="multiple",
                file_types=[".txt", ".pdf", ".doc", ".docx", ".md"],
                variant="primary",
                size="lg"
            )
            
            # Status display
            status_output = gr.Textbox(
                label="Upload Status",
                placeholder="Upload status will appear here...",
                interactive=False,
                lines=10
            )
    
    # Connect upload function
    upload_button.upload(
        fn=upload_file,
        inputs=upload_button,
        outputs=status_output,
        show_progress=True
    )

if __name__ == "__main__":
    demo.launch()