<template>
  <v-container class="fill-height">
    <v-row class="fill-height">
      <v-col cols="3" class="left-column d-flex flex-column align-center justify-center">
        <!-- Button for deleting and resetting page -->
        <v-btn color="red" @click="deleteAndReset">
          Delete and Reset Page
        </v-btn>

        <!-- Feedback message for deletion -->
        <div v-if="deleteFeedback">{{ deleteFeedback }}</div>

        <!-- File upload section -->
        <input type="file" ref="fileInput" @change="handleFileUpload" accept="application/pdf"/>
        <!-- Feedback message for file upload -->
        <div v-if="uploadFeedback">{{ uploadFeedback }}</div>
        <!-- Loading spinner for file upload -->
        <v-progress-circular v-if="loading" indeterminate color="blue"></v-progress-circular>
        <div v-if="uploadedFiles.length">
          <ul>
            <li v-for="(file, index) in uploadedFiles" :key="index">{{ file }}</li>
          </ul>
        </div>
      </v-col>

      <v-col cols="9" class="right-column d-flex flex-column justify-end pa-3">
        <v-form @submit.prevent="sendQuestion" class="px-3 ml-16 mr-16">
          <v-text-field
            v-model="question"
            class="text-input"
            append-icon="mdi-send"
            placeholder="Type your message..."
            solo
            hide-details
            dense
            @click:append="sendQuestion"
            @keyup.enter="sendQuestion"
          ></v-text-field>
        </v-form>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import axios from 'axios';

export default {
  name: 'FileManagement',
  data() {
    return {
      deleteFeedback: '',
      uploadFeedback: '',
      uploadedFiles: [],
      loading: false,
      fileInput: null,
      question: ''
    };
  },
  mounted() {
    this.fileInput = this.$refs.fileInput;
  },
  methods: {
    deleteAndReset() {
      axios.delete('http://localhost:8000/indexes?key=', {})
        .then(response => {
          if (response.status === 202) {
            this.deleteFeedback = 'Data deleted and page reset!';
            setTimeout(() => this.deleteFeedback = '', 5000);
            this.fileInput.value = ''; // Reset the file input
            this.uploadedFiles = []; // Clear the list of uploaded files
            this.uploadFeedback = ''; // Clear upload feedback
          }
        })
        .catch(error => {
          console.error('Error:', error);
          this.deleteFeedback = 'Failed to delete data.';
          setTimeout(() => this.deleteFeedback = '', 5000);
        });
    },
    handleFileUpload(event) {
      const file = event.target.files[0];
      if (file) {
        this.loading = true;
        this.uploadedFiles.push(file.name);
        const formData = new FormData();
        formData.append('file', file);
        axios.post('http://localhost:8000/upload-pdf?key=', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })
          .then(response => {
            this.loading = false;
            this.uploadFeedback = 'File uploaded successfully';
            setTimeout(() => this.uploadFeedback = '', 5000);
          })
          .catch(error => {
            console.error('Upload failed:', error);
            this.loading = false;
            this.uploadFeedback = 'Upload failed';
            setTimeout(() => this.uploadFeedback = '', 5000);
          });
      }
    },
    sendQuestion() {
      if (this.question.trim()) {
        axios.post('http://localhost:8000/question?key=', {
          question: this.question
        })
          .then(response => {
            // Handle the response from the server here
            console.log('Question sent:', response);
            this.question = ''; // Reset input after sending
          })
          .catch(error => {
            console.error('Error sending question:', error);
          });
      }
    }
  }
};
</script>

<style>
.left-column {
  background-color: #292929; /* Dark grey color for the left column */
  color: #ffffff; /* White text color */
  gap: 20px; /* Space between elements */
}

.right-column {
  background-color: #333333; /* Light grey color for the right column */
  color: #000000; /* Black text color */
}

/* Custom styling for the text input field to blend in */
.text-input {
  background-color: #333333; /* Slightly lighter than the column for contrast */
  color: #ffffff;
  border: none;
  border-radius: 25px;
}

/* Custom styling for the input field's outline on focus */
.text-input::placeholder {
  color: #ffffff;
}

/* Vuetify's v-text-field--solo can sometimes apply a different background */
.v-text-field--solo{
  background-color: #333333;
}

</style>
