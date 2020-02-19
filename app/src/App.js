import React from 'react';
import './App.css';
import Navigation from './Navigation.js'
import Container from '@material-ui/core/Container';
import Grid from '@material-ui/core/Grid';
import Button from '@material-ui/core/Button';
import Paper from '@material-ui/core/Paper';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';
import Image from 'react-bootstrap/Image';

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      option:0,  
      submitted: false,
      loading: false,
      hasImage: false,
      imageUrl: '',
      fileUrl: '',
      file:null,
      result: ""
    }
     
  }

  handleOptionSelected = option => {
    this.setState({
      option: option
    })
    if (this.state.submitted) {
      this.setState({ submitted: false })
    }
    if (this.state.loading) {
      this.setState({ loading: false })
    }
  }

  handleChange = (event) => {

    const value = event.target.value;
    const name = event.target.name;
    console.log(value+" AND "+name);
    var hasImage = this.state.hasImage;
    hasImage = value !== '';
    this.setState({
      hasImage,
      fileUrl: URL.createObjectURL(event.target.files[0]),
      file:event.target.files[0],
    });

    if (this.state.submitted) {
      this.setState({ submitted: false })
    }
    if (this.state.loading) {
      this.setState({ loading: false })
    }

    if (this.state.result) {
      this.setState({ result: "" })
    }

  }

onSubmit = (event) => {

    this.setState({ loading: true })
    this.setState({ submitted: true });
    setTimeout(() => {
      this.setState({ loading: false })
    }, 2000);

    const file = this.state.file;
    console.log(" onSubmit ");

    fetch('http://172.17.0.5:5000/classify', 
      {
        method: 'POST',
        body: file,
        headers: {
        'Content-Type': 'application/octet-stream',
      }

      },  )
      .then(response => response.json())
      .then(response => {
        console.log(response)
        this.setState({
          result: response.predictions
        });
      });

}


  render() {

  return (
    <div className="App">
      <Container>
      <h1 class="title"> Document Image Classification</h1>
      <Navigation 
      option = {this.state.option}
      onSelect = {this.handleOptionSelected}
      handleChange = {this.handleChange}
      onSubmit = {this.onSubmit}
      imageUrl = {this.state.imageUrl}
    
      />

      {this.state.loading && (<h2>Processing...</h2>)}
         
         {this.state.submitted && !this.state.loading && (
            <Grid container spacing={3}>
              <Grid item xs={12}>

              <Paper>
                  <Table>

                    <TableBody>

                        <TableRow>
                          <TableCell component="th" scope="row">
                            <Image height={600} width={500} src={this.state.fileUrl} />
                            {this.state.result!=""&&this.state.result.map(item => (
                            <h3>{item.label} <spam style={{color:"blue"}}>{item.prob}%</spam></h3>
                            ))}
                            
                          </TableCell>
                        </TableRow>
                    </TableBody>
        </Table>
                </Paper>
              </Grid>
            </Grid>
          )}
      </Container>
      </div>


   
  );
}
}

export default App;
