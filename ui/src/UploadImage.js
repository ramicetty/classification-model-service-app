import React from 'react';
import Grid from '@material-ui/core/Grid';
import Button from '@material-ui/core/Button';
import Paper from '@material-ui/core/Paper';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';
import Image from 'react-bootstrap/Image';


export default props => {

	const hChange = (e) => props.handleChange(e)
	const documentSubmit = (e) => props.onSubmit(e)



	return (
		<>

		<form noValidate autoComplete="off" onSubmit={documentSubmit}>

		<Grid container spacing={3}>
              <Grid item xs={5}>
              <input type = "file" name="image"
              value={props.imageUrl}
              onChange={hChange} />  
              </Grid>
              <Grid item  xs={2} id='upload_button' >
                  <Button color="primary" variant="contained" type="submit">Submit</Button>
              </Grid>
        </Grid>
        </form>
         </>
         )
}